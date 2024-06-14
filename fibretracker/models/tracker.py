'''
Functions for tracking fibers in 3D volumes. The tracking is done by tracking fibers in the z-direction.

'''

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

from .detector import get_fibre_coords

class TrackPoints:
    def __init__(self, max_jump=5, max_skip=5, momentum=0.1, track_min_length=3):
        '''Initialization of the fiber tracker. The fiber tracker tracks fibers in the z-direction.

        Args:
            max_jump (float, optional): Maximum distance between detected points in two consecutive frames. Threshold in pixels.
            max_skip (int, optional): Maximum number of frames along one track where no points are detected.
            momentum (float, optional): Parameter in the range [0,1] that gives momentum to the tracking direction.
            track_min_length (int, optional): Minimum number of points in a track.

        Returns:
            None

        '''

        self.max_jump = max_jump**2 # not move more than max_jump pixels (using squared distance)
        if self.max_jump < 1: # should be at least 1 pixel
            self.max_jump = 1
        
        self.max_skip = max_skip # maximum number of slides that can be skipped in a track. If set to 0, then no slides can be skipped.
        if self.max_skip < 0:
            self.max_skip = 0

        self.momentum = momentum # direction momentum that must be between 0 and 1
        if self.momentum < 0:
            self.momentum = 0
        elif self.momentum > 1:
            self.momentum = 1
        
        self.track_min_length = track_min_length # minimum length of tracks that should be at least 1
        if self.track_min_length < 1:
            self.track_min_length = 1

        
    def __call__(self, coords):
        '''Call function for FiberTracker

        Args:
            coords (list): List of numpy arrays with row and column indices of detected points. One per slice, which
                            means that z is gives as the index of the list.
            
        Returns:
            tracks (list): List of numpy arrays each containing coordinates of tracked fibers.
            tracks_all (list): List of all tracked fibers.
        
        '''
        r = [coord[:,0] for coord in coords]
        c = [coord[:,1] for coord in coords]
        return self.track_fibers(r, c)
 
    def get_dist(self, ra, ca, rb, cb):
        '''Computes a 2D distance array between row and column coordinates in set a (ra, ca) and set b (rb, cb) 

        Args:
            ra (np.ndarray): 1D array of row coordinates of point a.
            ca (np.ndarray): 1D array of column coordinates of point a.
            rb (np.ndarray): 1D array of row coordinates of point b.
            cb (np.ndarray): 1D array of column coordinates of point b.

        Returns:
            dist (np.ndarray): n_a x n_b 2D euclidean distance array between the two point sets.


        '''
        ra = np.array(ra)
        ca = np.array(ca)
        rb = np.array(rb)
        cb = np.array(cb)

        dist = ((np.outer(ra, np.ones((1,len(rb)))) - np.outer(np.ones((len(ra),1)), rb))**2 + 
                (np.outer(ca, np.ones((1,len(cb)))) - np.outer(np.ones((len(ca),1)), cb))**2)
        return dist
    
    
    def swap_place(self, tr, id_first, id_second):

        '''Swaps the place of two elements in a list.

        Args:
            tr (list): List of elements.
            id_first (int): Index of first element.
            id_second (int): Index of second element.

        Returns:
            list: Swapped list of elements.

        '''

        tmp = tr[id_first]
        tr[id_first] = tr[id_second]
        tr[id_second] = tmp
        return tr
    

    def track_fibers(self, r, c):
        '''Tracks fibers throughout the volume

        Args:
            r (list): List of numpy arrays with row coordinates of detected points.
            c (list): List of numpy arrays with column coordinates of detected points.

        Returns:
            tracks (list): List of numpy arrays each containing coordinates of tracked fibers.
            tracks_all (list): List of all tracked fibers.


        '''
        
        tracks_all = [] # Coordinates
        ntr_ct = [] # count of not found points
        
        # initialize tracks (row, col, layer, drow, dcol) and counter for tracks
        for ra, ca in zip(r[0], c[0]):
            tracks_all.append([(ra, ca, 0, 0, 0)])
            ntr_ct.append(0)

        coord_r = r[0].copy()
        coord_c = c[0].copy()
        
        nf_counter = 0 # counter for not found points
        for i in range(1, len(r)): # Loop over slices
            
            # Get distance from previous slice to current slice
            d = self.get_dist(coord_r, coord_c, r[i], c[i])
            
            id_from = d.argmin(axis=0) # id of min distance (previous slide)
            id_to = d.argmin(axis=1) # id of min distance (current slide)
            
            d_from = d.min(axis=0) # min distance (previous slide)
                
            id_match_from = id_to[id_from] # matched id previous to current slide
            idx = id_match_from == np.arange(len(id_from)) # look up coordinates (id of matched points)
            for j in range(len(idx)): 
                if idx[j] and d_from[j] < self.max_jump: # if matched and distance is less than max_jump
                    drow = (self.momentum*(r[i][j] - tracks_all[id_from[j] + nf_counter][-1][0]) + 
                            (1-self.momentum)*tracks_all[id_from[j] + nf_counter][-1][3]) # row direction
                    dcol = (self.momentum*(c[i][j] - tracks_all[id_from[j] + nf_counter][-1][1]) +
                            (1-self.momentum)*tracks_all[id_from[j] + nf_counter][-1][4]) # column direction
                    tracks_all[id_from[j] + nf_counter].append((r[i][j], c[i][j], i, drow, dcol)) # add track point to the matched track
                else:
                    tracks_all.append([(r[i][j], c[i][j], i, 0, 0)]) # start new track
                    ntr_ct.append(0) # reset counter for not found points for new track
                    
            not_matched = np.ones(len(coord_r), dtype=int)
            not_matched[id_from] = 0
            for j in range(len(not_matched)):
                if not_matched[j]:
                    ntr_ct[j + nf_counter] += 1
            
            coord_r = []
            coord_c = []
                        
            for j in range(nf_counter, len(tracks_all)):
                if ntr_ct[j] > self.max_skip:
                    ntr_ct = self.swap_place(ntr_ct, j, nf_counter)
                    tracks_all = self.swap_place(tracks_all, j, nf_counter) 
                    nf_counter += 1
            
            for j in range(nf_counter, len(tracks_all)):
                coord_r.append(tracks_all[j][-1][0] + (i-tracks_all[j][-1][2])*tracks_all[j][-1][3]) # update row coordinates of previous slice for next iteration with momentum
                coord_c.append(tracks_all[j][-1][1] + (i-tracks_all[j][-1][2])*tracks_all[j][-1][4]) # update column coordinates of previous slice for next iteration with momentum
            if i%10 == 9:
                print(f'Fibre tracking: slice {i+1} out of {len(r)}', end='\r')
        print(' ' * len(f'Detecting coordinates - slice: {i+1}/{len(r)}'), end='\r')
        
        tracks = []
        for track in tracks_all:
            track_len = 0
            track_arr = np.stack(track)
            for i in range(1, len(track)):
                track_len += np.linalg.norm(track_arr[i]-track_arr[i-1])
            if track_len > self.track_min_length:
                track_arr = np.stack(track)
                tracks.append(track_arr[:,:3]) 
        
        return tracks, tracks_all
    
    def fill_track(self, track):
        '''Fills in missing points in a track by linear interpolation. 

        Args:
            track (np.ndarray): Single track with shape (n_points, 3)
            
        Returns:
            t (np.ndarray): Filled track.

        '''
        n = int(track[-1,2] - track[0,2] + 1)
        t = np.zeros((n,3))
        ct = 1
        t[0] = track[0]
        for i in range(1,n):
            if track[ct,2] == i + track[0,2]:
                t[i] = track[ct]
                ct += 1
            else:
                nom = (track[ct,2] - track[ct-1,2])
                den1 = (track[ct,2] - track[0,2]) - i
                den2 = i - (track[ct-1,2] - track[0,2])
                w1 = den1/nom
                w2 = den2/nom
                t[i,0:2] = track[ct-1,0:2]*w1 + track[ct,0:2]*w2
                t[i,2] = i + track[0,2]
        return t.astype(int)
    
    def fill_tracks(self, tracks):
        '''Fills in missing points in a list of tracks by linear interpolation.

        Args:
            tracks (list): list of numpy arrays (npoints, 3)

        Returns:
            tracks_filled (list): List of filled tracks.

        '''

        tracks_filled = []
        for track in tracks:
            tracks_filled.append(self.fill_track(track))
        return tracks_filled


def track_fibres(
        vol: Optional[np.ndarray]=None,
        max_jump: int=5, 
        max_skip: int=5, 
        momentum: float=0.1, 
        track_min_length: int=5,
        coords: Optional[List[np.ndarray]]=None,
        std: float=2.5,
        min_distance: int=5,
        threshold_abs: float=0.5,
        weighted_avg: bool=False,
        window_size: int=10,
        apply_filter: bool=False,
        smoothtrack_gaussian: bool=False,
        sigma: float=3,
        smoothtrack_watershed: bool=False,
        threshold: Optional[float]=None
        ):
    '''Tracks fibers throughout the volume

        Args:
            vol (np.ndarray, optional): 3D volume.
            max_jump (int, optional): Maximum distance between detected points in two consecutive frames. Threshold in pixels.
            max_skip (int, optional): Maximum number of frames along one track where no points are detected.
            momentum (float, optional): Parameter in the range [0;1] that gives momentum to the tracking direction.
            track_min_length (int, optional): Minimum number of points in a track.
            coords (list, optional): List of numpy arrays with row and column indices of detected points. One per slice, which
                                        means that z is gives as the index of the list.
            std (float, optional): Standard deviation of the Gaussian filter.
            min_distance (int, optional): Minimum distance between fibres.
            threshold_abs (float, optional): Threshold value for the peak from the background.
            weighted_avg (bool, optional): Whether to apply weighted average to the detected coordinates.
            window_size (int, optional): Size of the neighbourhood window around the peak.
            apply_filter (bool, optional): Whether to apply Gaussian filter to the window.
            smoothtrack_gaussian (bool, optional): Whether to smooth tracks using Gaussian.
            sigma (float, optional): Sigma value for Gaussian filter.
            smoothtrack_watershed (bool, optional): Whether to smooth tracks using watershed.
            threshold (float, optional): Threshold value for watershed.

        Returns:
            tracks (List[np.ndarray]): List of arrays of shape (n_points, 3) - each list contains coordinates of tracked fibers.

        Example:
            ```python
            import fibretracker as ft
            
            vol = ft.io.load("path/to/volume.txm")
            v = ft.io.normalize(vol)
            coords = ft.models.get_fibre_coords(v)
            tracks = ft.models.track_fibres(coords=coords)
            ```
            ```python
            import fibretracker as ft
            
            vol = ft.io.load("path/to/volume.txm")
            v = ft.io.normalize(vol)
            coords = ft.models.get_fibre_coords(v)
            tracks = ft.models.track_fibres(vol=v, coords=coords, smoothtrack_gaussian=True, sigma=3) # Smoothened tracks using Gaussian
            ```


    '''

    fib_tracker = TrackPoints(max_jump=max_jump, max_skip=max_skip,
                        momentum=momentum, track_min_length=track_min_length)
    
    if coords is None:
        coords = get_fibre_coords(vol, std=std, min_distance=min_distance, threshold_abs=threshold_abs, 
                                weighted_avg=weighted_avg, window_size=window_size, apply_filter=apply_filter)
    
    tracks, _ = fib_tracker(coords)
    if smoothtrack_gaussian:
        print('Smoothing tracks using Gaussian')
        tracks_smooth = []
        for track in tracks:
            tracks_smooth.append(gaussian_filter1d(track, sigma=sigma, axis=0))
        
        return tracks_smooth

    elif smoothtrack_watershed and vol is not None:
        print('Smoothing tracks using watershed...')
        tracks_filled = fib_tracker.fill_tracks(tracks)
        if threshold is None:
            threshold = threshold_otsu(vol[len(vol)//2])
        V_thres = vol > threshold
        V_dist = -distance_transform_edt(V_thres)
        V_coords = np.zeros(vol.shape)
        for i, track in enumerate(tracks_filled):
            for point in track:
                V_coords[int(point[2]), int(point[1]), int(point[0])] = i + 1
        V_ws = watershed(V_dist, markers=V_coords.astype(int))*V_thres
        print('Watershed volume created.')
        n_fibers = V_ws.max()
        tracks_smooth = [[] for i in range(n_fibers)]
        
        for i, v_ws in enumerate(V_ws):
            props = regionprops(v_ws)
            for prop in props:
                tracks_smooth[prop.label-1].append(list(prop.centroid[::-1]) + [i])
            print(f'Smoothing tracks - iteration: {i+1}/{len(V_ws)}', end='\r')

        for i in range(n_fibers):
            tracks_smooth[i] = np.array(tracks_smooth[i])
        
        return tracks_smooth
    
    else:
        return tracks

