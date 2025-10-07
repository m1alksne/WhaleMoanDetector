# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:23:05 2024

@author: Michaela Alksne
"""

import numpy as np

def time_to_pixels(start_time, end_time, spec_width, window_size):
    """Map start and end times in seconds to pixel positions in the spectrogram."""
    pixels_per_second = spec_width / window_size
    xmin = int(np.floor(start_time * pixels_per_second))
    xmax = int(np.ceil(end_time * pixels_per_second))
    return xmin, xmax
