# Prepro-F25 – University of Michigan SunRISE Mission

**Authors:** Callen Fields, Aashi Mishra  

---

## Overview

This repository contains Python scripts for downloading, processing, and visualizing **eCallisto solar radio spectrograms**

## File Descriptions

### `one_day.py`

- **Purpose:** Download all recordings from a given eCallisto station for a specific day, concatenate them into a single array, and save as a `.npy` file.
- **Key Functions:**
  - `download_fits_from_gz(url: str) -> np.ndarray`: Downloads and reads a `.fit.gz` file into a NumPy array.
  - `circular_sort(files: List[str], offset: str) -> List[str]`: Sorts filenames by UTC time, starting from a given offset.
  - `one_day(station, year, month, day, time="000000")`: Downloads, orders, and concatenates spectrograms for a single day.
  - `extract_bursts(station, year, month, day)`: Collects a station's identified burst start/end times from labels given here: "https://soleil.i4ds.ch/solarradio/data/BurstLists/2010-yyyy_Monstein/"
  - `find_bursts(arr, burst_list, filename, results, current_idx)`: Determines the index into the concatenated spectrogram that bursts begin and end.
- **Usage:**

    `python one_day.py <station> <month> <day> <year> [start_time] [--save_burst_labels]`

- **Example:**

    `python one_day.py ALASKA-ANCHORAGE 5 13 2025 13000000`

    This saves a file called spec-ALASKA-ANCHORAGE-5-13-2025.npy.

### `plot_spectrogram.py`

- **Purpose:** Load a `.npy` spectrogram file and display it using matplotlib.
- **Key Function:**
    - `plot_spectrogram(big_array: np.ndarray, cmap="viridis")`: Plots a 2D spectrogram with frequency vs. time.

- **Usage:**

    `python plot_spectrogram.py <spectrogram_file_path> [labels_file_path]`

- **Example:**

    `python plot_spectrogram.py spec-ALASKA-ANCHORAGE-5-13-2025.npy`

### `compute_snr.py`

- **Purpose:** Calculate the signal-to-noise ratio of a solar spectrogram by computing the mean flux value of regions with and without a solar burst occurence.
- **Key Functions:**
  - `compute_snr(spectrogram, burst_labels)`: Compute SNR of a spectrogram given burst index ranges.
- **Usage:**

    `python compute_snr.py <spectrogram_file_path> <labels_file_path>`

- **Example:**

    `python compute_snr.py spec-MONGOLIA-UB-05-13-2025.npy labels-MONGOLIA-UB-05-13-2025.npy`

    This will output:  
    `Signal mean: 133.463`  
    `Noise mean: 133.636`   
    `SNR: -0.01 dB` 

### `AGBS.py`

- **Purpose:** Filter out noise in a solar spectrogram using Adaptive Gaussian Background Subtraction
- **Key Functions:**
  - `AGBS(spectrogram, seconds_window=60)`: Compute AGBS given a spectrogram and length of the rolling mean window
- **Usage:**

    `python AGBS.py <spectrogram_file_path>`

- **Example:**

    `python AGBS.py spec-MONGOLIA-UB-05-13-2025.npy`

    This will output the file:
    `spec-MONGOLIA-UB-05-13-2025-AGBS.npy`

### `AMF.py`

- **Purpose:** Filter out noise in a solar spectrogram using Adaptive Median Filtering
- **Key Functions:**
  - `AMF(spectrogram, radius=15)`: Compute AMF given a spectrogram and maximum kernel size
- **Usage:**

    `python AMF.py <spectrogram_file_path>`

- **Example:**

    `python AMF.py spec-MONGOLIA-UB-05-13-2025-AGBS.npy`

    This will output the file:
    `spec-MONGOLIA-UB-05-13-2025-AGBS-AMF.npy`

## Requirements

Install dependencies with:
`pip install -r requirements.txt`

