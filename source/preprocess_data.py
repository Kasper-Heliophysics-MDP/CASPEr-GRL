"""
===============================================================================
csv_to_npy.py

Author: Callen Fields
Edited: Michael Shepard
Date: 2025-11-23
Purpose:
    Convert a CSV containing time-frequency data into NumPy `.npy` and `.npz` files.
    The CSV must be generated using Radio-Sky Spectrograph - https://www.radiosky.com/specdownload.html

Features:
    - Optional linear interpolation to fill zero-valued gaps in the spectrogram
    - Optional preprocessing with Adaptive Gaussian Background Subtraction (AGBS)
      and Adaptive Median Filtering (AMF)
    - Resampling of time axis from 10 Hz → 4 Hz with aligned `times` array
    - Saves processed data as `.npy` and `.npz` for plotting or further analysis

Usage:
    python csv_to_npy.py <csv_file> [--fill_zeros] [--preprocess]

Example:
    python csv_to_npy.py data/my_spectrogram.csv --fill_zeros --preprocess
===============================================================================
"""
import config
import os
import numpy as np
import pandas as pd
from preprocess.AGBS import AGBS
from preprocess.AMF import AMF
from plot_npy import plot_npy

def fill_zero_gaps(data: np.ndarray) -> np.ndarray:
    """
    Fill zero-valued gaps in a 2D array (time x frequency)
    by linear interpolation along the time axis.
    """
    filled = data.copy()
    n_time, n_freq = filled.shape

    for i in range(n_freq):
        col = filled[:, i]
        if np.any(col != 0):
            s = pd.Series(col)
            filled[:, i] = (
                s.replace(0, np.nan)
                 .interpolate(limit_direction="both")
                 .to_numpy()
            )
    return filled

def preprocess(data: np.ndarray) -> np.ndarray:
    """
    Run preprocessing tools developed in Prepro-F25 repo
    """
    data = AGBS(data)
    data = AMF(data)
    return data


def process_data(output_dir: str, filename: str, do_interp: bool = False, prepro: bool = False):
    """
    Load a NPY with 'Datetime' column and frequency columns,
    optionally fill zero gaps, and save as .npy + .npz.
    """
    # Extract frequency columns (convert from string to float)
    values = np.load(filename)

    if prepro:
        print(" \tPreprocessing data...")
        values = preprocess(values)

    # Save just the values to .npy
    np.save(f"{output_dir}/{os.path.basename(filename)}", values)
    print(f" \tSaved data to {os.path.basename(filename)}.npy (shape={values.shape})\n")

def preprocess_npy():
    source_dir = config.NPY_DUMP_PRP
    fill_zero  = config.FILL_ZERO_PRP
    preprocess_data = config.PREPROCESS_PRP
    output_dir = config.CLEANED_NPY_PRP

    # Collect all npy files
    npy_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".npy"):
                npy_files.append(os.path.join(root, file))

    for file_name in npy_files:
        plot_npy(f"{source_dir}/{os.path.basename(file_name)}", True)
        process_data(output_dir, file_name, do_interp=fill_zero, prepro=preprocess_data)
        plot_npy(f"{output_dir}/{os.path.basename(file_name)}", True)

if __name__ == "__main__":
    print(f"*\tBeginning Preprocessing...")
    preprocess_npy()


