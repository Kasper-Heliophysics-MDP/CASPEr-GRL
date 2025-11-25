'''
===============================================================================
File: AGBS.py
Author: Callen Fields (fcallen@umich.edu), Aashi Mishra (aashim@umich.edu)
Date: 2025-10-6
Group: University of Michigan SunRISE Mission

Description:
Performs adaptive gaussian background subtraction on a spectrogram
For each band, calculate mean and std over the whole file. At each time
step, if the value is above 3*std, subtract the long term mean. If not,
subtract a rolling local mean.

https://www.swsc-journal.org/articles/swsc/pdf/2018/01/swsc170092.pdf
===============================================================================
'''
import numpy as np
import sys

UNITS_PER_SECOND = 4

def AGBS(spec, seconds_window=60):
    """
    Adaptive Gaussian Background Subtraction
    Local mean is calculated via convolution with all ones.
    
    Args:
        spec (np.Arr): Frequency x Time spectrogram
        seconds_window (int): length of the rolling mean window in seconds 
        month (int):
        day (int): The date to look at
        time (str): UTC time that designates the beginning of the day in "HHMMSS" format
        
    Returns:
        numpy arr: processed 2D spectrogram
    """    
    if spec.ndim != 2:
        raise ValueError("Spectrogram must be a 2D array (freq x time).")

    n_freqs, n_time = spec.shape
    processed = np.zeros_like(spec)

     # Precompute rolling window size
    window_size = int(seconds_window * UNITS_PER_SECOND)
    half_window = window_size // 2

    for f in range(n_freqs):
        band = spec[f, :]
        mean_all = np.mean(band)
        std_all = np.std(band)
        threshold = mean_all + 3 * std_all

        # Compute rolling mean efficiently using convolution
        kernel = np.ones(window_size) / window_size
        rolling_mean = np.convolve(band, kernel, mode='same')

        # Masks for values above and below the threshold
        mask_high = band > threshold

        # For values in the quiet background: subtract local rolling mean
        processed[f, :] = band - rolling_mean

        # For values in a burst: subtract global mean instead
        processed[f, mask_high] = band[mask_high] - mean_all

    return processed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python AGBS.py <spectrogram.npy>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".npy", "-AGBS.npy")
    spec = np.load(input_file)

    processed_spec = AGBS(spec)
    np.save(output_file, processed_spec)

    print(f"Processed spectrogram saved to {output_file}")