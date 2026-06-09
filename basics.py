# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
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
    
    # return r_bars, g_bars, b_bars
    return [0], [0], [0]



####

### All other basic functions

####