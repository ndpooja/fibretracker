
import scipy
import numpy as np 
import skimage

from typing import Optional


def gaussian_filter(
        std: float
        ):
    ''' Generate a 1D Gaussian filter and its derivatives

    Args: std: float, standard deviation of the Gaussian filter
    Returns: g: np.ndarray, 1D Gaussian filter
             dg: np.ndarray, derivative of the Gaussian filter
             ddg: np.ndarray, second derivative of the Gaussian filter

    '''
    x = np.arange(-np.ceil(5*std), np.ceil(5*std) + 1)[:,None]
    g = np.exp(-x**2/(2*std**2))
    g /= np.sum(g)
    dg = -x/std**2 * g
    ddg = -g/std**2 -x/std**2 * dg
    return g, dg, ddg

def blob_detector(
        im: np.ndarray, 
        std: float=2.5, 
        min_distance: int=3, 
        threshold_abs: float=0.4
        ):
    ''' Predict the coordinates of the peaks in the image using blob detector
    Args: 
        im: np.ndarray, input image
        s: float, standard deviation of the Gaussian filter
        min_distance: int, minimum distance between peaks
        threshold_abs: float, minimum intensity of peaks
    Returns: 
        pred_coords: np.ndarray, predicted coordinates of the peaks

    '''
    g = gaussian_filter(std)[0]
    im_g = scipy.ndimage.convolve(scipy.ndimage.convolve(im, g), g.T)
    pred_coords = skimage.feature.peak_local_max(im_g, min_distance=min_distance, threshold_abs=threshold_abs)
    return pred_coords

def get_coords(
        vol: np.ndarray, 
        std: float=2.5, 
        min_distance: int=3, 
        threshold_abs: float=0.4
        ):
    ''' Get the coordinates of the peaks in the volume

    Args:
        vol: np.ndarray, input volume
        std: float, standard deviation of the Gaussian filter
        min_distance: int, minimum distance between peaks
        threshold_abs: float, minimum intensity of peaks
    Returns:
        coords: np.ndarray, coordinates of the peaks

    '''
    coords = []
    for i, im in enumerate(vol):
        coord = blob_detector(im, std=std, min_distance=min_distance, threshold_abs=threshold_abs)
        coords.append(np.stack([coord[:,1], coord[:,0], np.ones(len(coord)) * i], axis=1))
        print(f'Detecting coordinates - slice: {i+1}/{len(vol)}', end='\r')
    print(' ' * len(f'Detecting coordinates - slice: {i+1}/{len(vol)}'), end='\r')
    return coords
    
def weighted_average_coords(
        pred_coord: np.ndarray, 
        im: np.ndarray, 
        window_size: int = 10,
        apply_filter: bool = False,
        std: Optional[float] = None
        ):
    ''' Compute the weighted average of the coordinates of the peaks

    Args: 
        pred_coord: np.ndarray, predicted coordinates of the peaks
        im: np.ndarray, input image
        window_size: int, size of the window around the peak
        apply_filter: bool, whether to apply Gaussian filter to the window
        std: Optional[float], standard deviation of the Gaussian filter
    Returns:
        coords: np.ndarray, weighted average of the coordinates of the peaks

    '''
    coords = []
    for coord in pred_coord:
        x, y = coord
        window = im[x-window_size//2:x+window_size//2+1, y-window_size//2:y+window_size//2+1]
        # Apply Gaussian filter to the window
        if apply_filter:
            if std is not None:
                g = gaussian_filter(std)[0]
            else:
                g = gaussian_filter(std=2.5)[0]
            window = scipy.ndimage.convolve(scipy.ndimage.convolve(window, g), g.T)
        x_coords, y_coords = np.meshgrid(range(x-window_size//2, x+window_size//2+1), range(y-window_size//2, y+window_size//2+1))
        weighted_x = np.sum(window * x_coords) / np.sum(window)
        weighted_y = np.sum(window * y_coords) / np.sum(window)
        coords.append([weighted_x, weighted_y])
    return np.array(coords)