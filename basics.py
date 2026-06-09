# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
import numpy as np
from numba import njit # conda install numba

@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''
    height = np_img.shape[0]
    width = np_img.shape[1]
    r_hist = np.zeros(256, np.int64)
    g_hist = np.zeros(256, np.int64)
    b_hist = np.zeros(256, np.int64)

    for y in range(height):
        for x in range(width):
            pixel = np_img[y, x]
            b_hist[pixel[0]] += 1
            g_hist[pixel[1]] += 1
            r_hist[pixel[2]] += 1

    max_value = 1
    for i in range(256):
        if r_hist[i] > max_value:
            max_value = r_hist[i]
        if g_hist[i] > max_value:
            max_value = g_hist[i]
        if b_hist[i] > max_value:
            max_value = b_hist[i]

    r_bars = np.empty(256, np.float64)
    g_bars = np.empty(256, np.float64)
    b_bars = np.empty(256, np.float64)
    scale = 3.0 / max_value

    for i in range(256):
        r_bars[i] = r_hist[i] * scale
        g_bars[i] = g_hist[i] * scale
        b_bars[i] = b_hist[i] * scale

    return r_bars, g_bars, b_bars



####

### All other basic functions

####