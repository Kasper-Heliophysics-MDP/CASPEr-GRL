"""
===============================================================================
plot_npy.py

Author: Callen Fields
Date: 2025-11-23
Purpose:
    Plot a frequency vs. time spectrogram from a `.npy` file. Optionally uses 
    metadata from an accompanying `.npz` file to label the time and frequency 
    axes correctly. Can also save the plot as a `.png` file.

Features:
    - Loads a `.npy` spectrogram file
    - Reads optional `.npz` metadata for proper time and frequency axes
    - Displays the spectrogram with a colorbar
    - Optionally saves the figure as a `.png` in the same directory as the input file

Usage:
    python plot_npy.py <file_prefix> [--save]

Example:
    python plot_npy.py data/my_spectrogram --save
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_npy(filename: str, save: bool = False):
    """
    Plot frequency vs time spectrogram from npy file
    """

    values = np.load(filename)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Amplitude / Value")

    # X-axis (Time)
    times = None
    if times is not None:
        num_ticks = min(6, len(times))
        tick_indices = np.linspace(0, len(times) - 1, num_ticks, dtype=int)
        time_labels = [np.datetime_as_string(t, unit="s").split("T")[-1] for t in times[tick_indices]]
        plt.xticks(tick_indices, time_labels, rotation=30, ha="right")
        plt.xlabel("Time (UTC)")
    else:
        plt.xlabel("Time Index")

    # Y-axis (Frequency)
    freqs = None
    if freqs is not None:
        plt.ylabel("Frequency (Hz)")
    else:
        plt.ylabel("Frequency Index")

    plt.title("Frequency-Time Plot")
    plt.tight_layout()
    if save:
        pass
    plt.show()

def main():
    plot_npy("/_cleaned_npy/240318204001-UofM.npy", save=False)


if __name__ == "__main__":
    main()


