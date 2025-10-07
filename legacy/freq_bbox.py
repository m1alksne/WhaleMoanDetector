# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:24:23 2024

@author: Michaela Alksne
"""
import numpy as np

def freq_to_pixels(low_f, high_f, spec_height, sr, n_fft):
    """Map frequencies to pixel positions in the spectrogram, accounting for the inversion."""
    # The frequency range we are mapping into (10 Hz to 150 Hz)
    low_freq_bound = 10
    high_freq_bound = 150

    # Calculate the frequency resolution of the spectrogram
    freq_resolution = (sr / 2) / (n_fft // 2)

    # How many Hz each frequency bin in the spectrogram represents
    freq_per_bin = freq_resolution

    # Calculate the total number of frequency bins
    total_freq_bins = spec_height

    # Calculate pixel position for the low and high frequencies, taking into account the inversion
    # The pixel position is calculated from the bottom (low frequencies) because of the inversion
    ymin = total_freq_bins - int(np.round((high_f - low_freq_bound) / freq_per_bin))
    ymax = total_freq_bins - int(np.round((low_f - low_freq_bound) / freq_per_bin))

    # Ensure ymin and ymax are within the bounds of the spectrogram's height
    ymin = max(0, ymin)
    ymax = min(spec_height - 1, ymax)  # Ensure ymax does not exceed the spectrogram's height
    
    return ymin, ymax