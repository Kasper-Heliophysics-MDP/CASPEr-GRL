"""
===============================================================================
locate_bursts.py

Author: Callen Fields
Date: 2025-11-23
Purpose:
    Detect and visually confirm bursts in a 2D spectrogram. This script first 
    identifies candidate bursts using a multi-level 3-sigma detection, then 
    provides an interactive viewer for the user to confirm real bursts. Confirmed 
    bursts are saved as `.npy` files for further analysis.

Features:
    - Loads spectrograms from `.npy` files with associated `.npz` metadata
      (`times` and `frequencies`)
    - Detects potential burst windows using binning and robust 3-sigma filtering
    - Adds padding around each burst window for context
    - Interactive Matplotlib viewer with checkboxes for selecting confirmed bursts
    - Saves confirmed bursts as individual `.npy` files in a specified output directory

Usage:
    python locate_bursts.py <file_prefix>

Example:
    python locate_bursts.py data/my_spectrogram
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import os
import sys
from cont_3sig import apply_robust_clip, bin_spectrogram, TIME_BIN_FACTOR

UNITS_PER_SECOND = 4
WINDOW_SIZE = 150*UNITS_PER_SECOND #each recording will be 5 minutes
DATA_DIR = "bursts" #where will this script save files to
WINDOW_PAD = 60*UNITS_PER_SECOND #add a minute to the beginning and end of each burst window

def select_and_save_bursts(spec, meta, windows, file_prefix="date-station", output_dir="confirmed_bursts"):
    """
    Interactive viewer for candidate bursts. Each burst is shown in a 5-min window centered on the detected burst.
    User clicks on figures to mark which bursts are real, and confirmed ones are saved as .npy files.

    Args:
        spec_path (str): Path to .npy spectrogram file.
        windows (list[int]): time indices of burst starts and ends [(start, end),...].
        window_minutes (float): Width of window in minutes (default 5).
        output_dir (str): Directory to save selected bursts.
    """
    times = meta.get("times")
    freqs = meta.get("frequencies")
    print(times.shape)

    os.makedirs(output_dir, exist_ok=True)

    n_bursts = len(windows)
    confirmed = [False] * n_bursts

    # Create figure
    ncols = min(4, n_bursts)
    nrows = int(np.ceil(n_bursts / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    # Plot each burst
    burst_idx = 0
    for (start, end), ax in zip(windows, axes):

        # Extract window from spectrogram
        s_window = spec[:, start:end]
        t_window = times[start:end]

        print(start)
        print(end)
        print(times[start])
        print(times[end])

        im = ax.imshow(
            s_window,
            aspect="auto",
            origin="lower",
            extent=[0, len(t_window), freqs.min(), freqs.max()],
            cmap="viridis"
        )
        fig.colorbar(im, ax=ax, label="Amplitude / Value")

        # X-axis labels
        num_ticks = min(6, len(t_window))
        tick_indices = np.linspace(0, len(t_window) - 1, num_ticks, dtype=int)
        time_labels = [np.datetime_as_string(t, unit="s").split("T")[-1] for t in t_window[tick_indices]]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(time_labels, rotation=30, ha="right")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Burst idx {burst_idx}")
        burst_idx += 1

    # Checkboxes
    rax = plt.axes([0.92, 0.1, 0.07, 0.8])
    labels = [f"{i}" for i in range(n_bursts)]
    check = CheckButtons(rax, labels, confirmed)

    def toggle(label):
        idx = labels.index(label)
        confirmed[idx] = not confirmed[idx]

    check.on_clicked(toggle)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    for (start, end), sel in zip(windows, confirmed):
        if sel:
            # Extract padded spectrogram region
            s_window = spec[:, start:end]

            # Use padded start for filename time
            t_start = times[start]
            time_str = np.datetime_as_string(t_start, unit="s").split("T")[-1].replace(":", "")
            save_name = f"{file_prefix}-{time_str}.npy"
            save_path = os.path.join(output_dir, save_name)

            np.save(save_path, s_window)
            print(f"✅ Saved confirmed burst: {save_path}")

    print(f"Done! Total confirmed bursts: {sum(confirmed)}")

def get_windows(spec):
    """
    Multiple levels of continuous 3-sigma detection to locate potential bursts
    
    Args:
        spec (np.Array): processed spectrogram
        
    Returns:
        windows (np.Array): array of starts and ends for each burst window
    """
    #credit Jenny Nam
    plt.imshow(spec, aspect='auto')
    plt.title("Original Spectrogram")
    plt.show()
    binned_spectrogram = bin_spectrogram(spec)
    outlier_mask = apply_robust_clip(binned_spectrogram)
    burst_data = np.full_like(binned_spectrogram, np.nan)
    burst_data[outlier_mask] = binned_spectrogram[outlier_mask]
    plt.imshow(burst_data, aspect='auto')
    plt.title("Processed Spectrogram")
    plt.show()
    

    #collapse to flux then mean filter
    burst_data = np.nan_to_num(burst_data, nan=0.0)
    flux_time = np.mean(burst_data, axis=0)  # collapse freqs → flux vs time
    kernel = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    rolling_mean = np.convolve(flux_time, kernel, mode='same')
    plt.plot(flux_time, label="Flux vs time")
    plt.plot(rolling_mean, label="Rolling mean")
    plt.title("Flux over time with rolling mean")
    plt.legend()
    plt.show()

    # Find rising and falling edges
    mean_all = np.mean(rolling_mean)
    std_all = np.std(rolling_mean)
    threshold = mean_all + 1 * std_all

    mask = rolling_mean > threshold
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]

    if mask[-1]:
        ends = np.r_[ends, len(rolling_mean)]

    windows = list(zip(starts, ends))
    padded_windows = []
    for s, e in windows:
        s_pad = max(0, s*TIME_BIN_FACTOR - WINDOW_PAD)
        e_pad = min(spec.shape[1], e*TIME_BIN_FACTOR + WINDOW_PAD)
        padded_windows.append((s_pad, e_pad))

    return padded_windows 

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python locate_bursts.py <file_prefix>")
        sys.exit(1)
    
    base_file = sys.argv[1]
    npy_file = base_file + ".npy"
    npz_file = base_file + ".npz"

    # Load .npy
    if os.path.exists(npy_file):
        try:
            spec = np.load(npy_file)
            print(f"✅ Loaded {npy_file}, shape = {spec.shape}")
        except Exception as e:
            print(f"❌ Error loading {npy_file}: {e}")
    else:
        print(f"❌ File not found: {npy_file}")

    # Load .npz
    if os.path.exists(npz_file):
        try:
            meta = np.load(npz_file)
            print(f"✅ Loaded {npz_file}, keys = {list(meta.keys())}")
        except Exception as e:
            print(f"❌ Error loading {npz_file}: {e}")
    else:
        print(f"❌ File not found: {npz_file}")

    windows = get_windows(spec)
    print(windows)

    select_and_save_bursts(
        spec,
        meta,
        windows,
        file_prefix=base_file,
        output_dir=DATA_DIR
    )