"""
===============================================================================
csv_to_npy.py

Author: Callen Fields
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
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess.AGBS import AGBS
from preprocess.AMF import AMF
from scipy.signal import resample


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



def csv_to_npy(filename: str, do_interp: bool = False, prepro: bool = False):
    """
    Load a CSV with 'Datetime' column and frequency columns,
    optionally fill zero gaps, and save as .npy + .npz.
    """
    base = Path(filename).with_suffix("")
    csv_file = base.with_suffix(".csv")
    df = pd.read_csv(csv_file)

    # Combine Date and Time into one datetime column
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")

    # Drop the original Date and Time columns
    df = df.drop(columns=["Date", "Time"])

    # Extract frequency columns (convert from string → float)
    frequencies = df.columns.drop("Datetime").astype(float).to_numpy()

    # Extract numeric data and time
    values = df.drop(columns="Datetime").to_numpy()
    times = df["Datetime"].to_numpy()

    if do_interp:
        print("Performing linear interpolation on zero gaps...")
        values = fill_zero_gaps(values)

    new_num_samples = int(values.shape[0] * 4 / 10)  # downsample from 10Hz → 4Hz
    values_resampled = resample(values, new_num_samples, axis=0)

    times_float = times.astype('datetime64[ns]').astype(np.int64)  # in ns
    times_resampled_float = np.linspace(times_float[0], times_float[-1], new_num_samples)
    times_resampled = times_resampled_float.astype('datetime64[ns]')


    values = values_resampled.T #transpose to be freq v time
    print(values.shape)

    if prepro:
        print("Preprocessing data...")
        values = preprocess(values)

    # Save just the values to .npy
    #np.save(base.with_suffix(".npy"), values) 
    print(f"✅ Saved data to {base.with_suffix('.npy')} (shape={values.shape})")

    # Optionally also save frequencies/times for later plotting
    np.savez(base.with_suffix('.npz'),
            times=times_resampled, frequencies=frequencies)

    print(f"Saved {base.with_suffix('.npy')} and {base.with_suffix('.npz')}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to NPY/NPZ for time-frequency data.")
    parser.add_argument("filename", type=str, help="Path to CSV file")
    parser.add_argument("-f", "--fill_zeros", action="store_true",
                        help="Enable linear interpolation over zero gaps")
    parser.add_argument("-p", "--preprocess", action="store_true",
                        help="Enable Adaptive Gaussian Background Subtraction and Adaptive Median Filtering")
    args = parser.parse_args()

    csv_to_npy(args.filename, do_interp=args.fill_zeros, prepro=args.preprocess)


if __name__ == "__main__":
    main()


