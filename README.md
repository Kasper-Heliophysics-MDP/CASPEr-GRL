# CASPEr-GRL
Computer Assisted Solar Phenomena Extractor for Ground Radio Lab

**Authors:** Callen Fields

---

## Overview
Ground Radio Lab data prep and metadata generation tool used to build datasets for ML applications

## Setup
This repo submodules the preprocessing tools repo. You MUST pull the submodule before using this repo
`git clone https://github.com/Kasper-Heliophysics-MDP/eCallisto-Burst-Grabber.git`
`cd eCallisto-Burst-Grabber`
`git submodule update --init --recursive`

Use `pip install -r requirements.txt` to import all required packages

## File Descriptions

### `cont_3sig.py`

- **Purpose:** Detect solar bursts in a 2D spectrogram using a highly efficient "bin and filter" method. The script downsamples the spectrogram into larger “super-pixels” and applies a fast Median Absolute Deviation (MAD) filter to identify bursts, achieving >100x speed compared to full-resolution filtering. Script developed by 2025 DSP subteam

- **Usage:**

    ```bash
    python cont_3sig.py
    ```

- **Parameters & Settings:**  
  - `FREQ_BIN_FACTOR` : Number of frequency channels to combine when binning (default: 5)  
  - `TIME_BIN_FACTOR` : Number of time steps to combine when binning (default: 10)  
  - `WINDOW_MINUTES` : Window size for MAD filter in minutes (default: 5)  
  - `SIGMA_THRESHOLD` : Threshold in number of sigmas to classify an outlier (default: 3)  
  - `STEPS_PER_MINUTE` : Original sampling rate in steps per minute (default: 190)  

- **Example:**  

    1. Update `INPUT_FILE` in the script to point to your `.npy` spectrogram file:

        ```python
        INPUT_FILE = "data/ALASKA-ANCHORAGE/ALASKA-ANCHORAGE_2024-07-20.npy"
        ```

    2. Run the script:

        ```bash
        python fast_robust_clip.py
        ```

- **Outputs:** The script generates the following files in the same directory as the input:

  - `*_binned_FxT.npy` : The binned spectrogram  
  - `*_binned_FxT_cleaned.npy` : Cleaned spectrogram with outliers replaced by NaN  
  - `*_binned_FxT_mask.npy` : Boolean mask of detected outliers  
  - `*_binned_FxT_bursts.npy` : Array highlighting only the bursts for visualization

This setup allows rapid detection and visualization of solar bursts while preserving the quiet regions for further analysis.

### `csv_to_npy.py`

- **Purpose:** Convert a spectrogram-style CSV file into NumPy `.npy` and `.npz` files for easy loading, plotting, or downstream processing. Includes optional zero-gap interpolation and preprocessing (AGBS + AMF). CSV file is generated using Radio-Sky Spectrograph: https://www.radiosky.com/specdownload.html

- **Usage:**  

    ```bash
    python csv_to_npy.py <csv_file> [--fill_zeros] [--preprocess]
    ```

- **Options:**  
  - `-f`, `--fill_zeros` : Fill zero-valued gaps in the spectrogram using linear interpolation  
  - `-p`, `--preprocess` : Apply Adaptive Gaussian Background Subtraction (AGBS) and Adaptive Median Filtering (AMF)

- **Example:**  

    ```bash
    python csv_to_npy.py data/my_spectrogram.csv --fill_zeros --preprocess

### `locate_bursts.py`

- **Purpose:** Detect and visually confirm bursts in a 2D spectrogram. The script first applies a multi-level 3-sigma detection to locate candidate bursts, then presents an interactive viewer where the user can select confirmed bursts. Selected bursts are saved as `.npy` files for further analysis.

- **Usage:**

    ```bash
    python locate_bursts.py <file_prefix>
    ```

- **Parameters & Settings:**  
  - `WINDOW_SIZE` : Default window size in samples for rolling mean (5 minutes)  
  - `UNITS_PER_SECOND` : Sampling rate used to convert minutes → samples (default: 4)  
  - `WINDOW_PAD` : Padding added to each burst window on both sides (default: 1 minute)  
  - `DATA_DIR` : Directory to save confirmed bursts (default: `bursts`)  
  - `TIME_BIN_FACTOR` : Factor used to scale burst windows according to binning  

### `plot_npy.py`

- **Purpose:** Plot a frequency vs. time spectrogram from a `.npy` file. Optionally uses metadata from an accompanying `.npz` file for proper time and frequency axes and can save the figure as a `.png`.

- **Usage:**

    ```bash
    python plot_npy.py <file_prefix> [--save]
    ```