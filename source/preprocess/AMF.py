'''
===============================================================================
File: AMF.py
Author: Callen Fields (fcallen@umich.edu), Aashi Mishra (aashim@umich.edu)
Date: 2025-10-6
Group: University of Michigan SunRISE Mission

Description:
Performs adaptive medium filtering over a given spectrogram. At each pixel,
adaptively forms a kernel based off the angle of least change and a maximum
pixel distance of 15. Replaces each pixel value with the median of this kernel.

https://www.swsc-journal.org/articles/swsc/pdf/2018/01/swsc170092.pdf
===============================================================================
'''
import numpy as np
from scipy.ndimage import sobel, map_coordinates
import sys
from tqdm import tqdm

def AMF(spec, radius=15, chunk_size=5000):
    """
    Adaptive Median Filtering
    Partial derivatives are calculated with a Sobel kernel.
    instead of using integer pixel coordinates, map_coordinates interpolates
    values between pixels to make the math work out better.
    For example, if you are at pixel (5, 5) and the angle is 30 degrees,
    the first pixel added to the kernel is (5 + (sqrt(3)/2), 5 + (1/2)).
    Bilinear interpolation is used to index into images at float coordinates.
    
    Args:
        spec (np.Arr): Frequency x Time spectrogram
        radius (int): maximum euclidean distance between the edge of the kernel and the starting pixel
        
    Returns:
        numpy arr: processed 2D spectrogram
    """    
    spec = spec.astype(np.float32, copy=False)

    # Gradients (Sobel)
    Gx = sobel(spec, axis=1, mode='reflect')
    Gy = sobel(spec, axis=0, mode='reflect')
    theta = np.arctan2(Gy, Gx) + np.pi / 2

    nrows, ncols = spec.shape
    filtered = np.zeros_like(spec)

    dx = np.cos(theta)
    dy = np.sin(theta)

    rows = np.arange(nrows)[:, None]
    cols = np.arange(ncols)[None, :]

    # Process in time chunks to avoid huge arrays
    for start in tqdm(range(0, ncols, chunk_size), desc="AMF filtering"):
        end = min(start + chunk_size, ncols)
        # We'll collect all offset samples for this chunk
        samples = []

        for k in range(-radius, radius + 1):
            rr = rows + k * dy[:, start:end]
            cc = cols[:, start:end] + k * dx[:, start:end]
            values = map_coordinates(spec, [rr, cc], order=1, mode='reflect')
            samples.append(values)

        # Stack along small third dimension
        stack = np.stack(samples, axis=-1)
        filtered[:, start:end] = np.median(stack, axis=-1)
        del stack, samples  # free memory

    return filtered

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python AMF.py <spectrogram.npy>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".npy", "-AMF.npy")
    spec = np.load(input_file)

    processed_spec = AMF(spec)
    np.save(output_file, processed_spec)

    print(f"Processed spectrogram saved to {output_file}")