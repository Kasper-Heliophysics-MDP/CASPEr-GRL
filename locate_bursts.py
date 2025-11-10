import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
import os
from pathlib import Path

UNITS_PER_SECOND = 10
WINDOW_SIZE = 150*UNITS_PER_SECOND #each recording will be 5 minutes
DATA_DIR = "bursts" #where will this script save files to

def show_burst_windows(spec_path, burst_indices, output_dir="confirmed_bursts", window_minutes=5):
    """
    Interactive viewer for candidate bursts. Each burst is shown in a 5-min window centered on the detected burst.
    User clicks on figures to mark which bursts are real, and confirmed ones are saved as .npy files.

    Args:
        spec_path (str): Path to .npy spectrogram file.
        burst_indices (list[int]): Time indices (in columns) for potential bursts.
        window_minutes (float): Width of window in minutes (default 5).
        output_dir (str): Directory to save selected bursts.
    """
    base = Path(spec_path).with_suffix("")
    npy_path = base.with_suffix(".npy")
    meta_path = base.with_suffix(".npz")

    spec = np.load(npy_path)
    if meta_path.exists():
        meta = np.load(meta_path, allow_pickle=True)
        times = meta.get("times")
        freqs = meta.get("frequencies")
    else:
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Parse file name: YYMMDD-station-name.npy
    filename_parts = base.name.split("-")
    if len(filename_parts) < 2:
        raise ValueError("Filename must be of the form YYMMDD-station-name.npy")
    date_str = filename_parts[0]
    station_name = "-".join(filename_parts[1:])

    # Compute time step per column (in minutes)
    dt = (times[1] - times[0]) / np.timedelta64(1, "m")
    half_window = int((window_minutes / 2) / dt)

    n_bursts = len(burst_indices)
    confirmed = [False] * n_bursts

    # Create figure
    fig, axes = plt.subplots(n_bursts, 1, figsize=(12, 3 * n_bursts))
    if n_bursts == 1:
        axes = [axes]

    # Plot each burst
    for i, ax in enumerate(axes):
        center = burst_indices[i]
        start = max(0, center - half_window)
        end = min(spec.shape[1], center + half_window)
        s_window = spec[:, start:end]
        t_window = times[start:end]

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
        ax.set_title(f"Burst idx {center}")

    # Checkboxes
    rax = plt.axes([0.92, 0.1, 0.07, 0.8])
    labels = [f"{burst_indices[i]}" for i in range(n_bursts)]
    check = CheckButtons(rax, labels, confirmed)

    def toggle(label):
        idx = labels.index(label)
        confirmed[idx] = not confirmed[idx]

    check.on_clicked(toggle)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    # Save confirmed bursts
    for i, sel in enumerate(confirmed):
        if sel:
            center = burst_indices[i]
            start = max(0, center - half_window)
            end = min(spec.shape[1], center + half_window)
            s_window = spec[:, start:end]

            # Get start time for filename
            t_start = times[start]
            time_str = np.datetime_as_string(t_start, unit="s").split("T")[-1].replace(":", "")
            save_name = f"{date_str}-{time_str}-{station_name}.npy"
            save_path = os.path.join(output_dir, save_name)

            np.save(save_path, s_window)
            print(f"✅ Saved confirmed burst: {save_path}")

    print(f"Done! Total confirmed bursts: {sum(confirmed)}")

def get_high_values(spec):
    """
    Multiple levels of continuous 3-sigma detection to locate potential bursts
    
    Args:
        spec (np.Array): processed spectrogram
        
    Returns:
        burst_centers (np.Array): array of indices that determine the center of each burst
    """
    n_freqs, n_time = spec.shape
    for f in range(n_freqs):

        #3-sigma over each band
        band = spec[f, :]

        mean_all = np.mean(band)
        std_all = np.std(band)
        threshold = mean_all + 3 * std_all
        mask_high = band > threshold

        spec[f, :] = band * mask_high

    #collapse to flux then mean filter
    flux_time = spec.mean(axis=0)  # collapse freqs → flux vs time
    kernel = np.ones(2*WINDOW_SIZE) / 2*WINDOW_SIZE
    rolling_mean = np.convolve(flux_time, kernel, mode='same')
    plt.plot(rolling_mean)
    plt.show()

    #3-sigma over the filtered flux data
    mean_all = np.mean(rolling_mean)
    std_all = np.std(rolling_mean)
    threshold = mean_all + 3 * std_all
    mask_high = rolling_mean > threshold
    indices = np.where(mask_high)[0]

    #when you see a high value, the next WINDOW_SIZE indices below to that burst
    burst_centers = []
    current_index = -WINDOW_SIZE-1 
    for i in indices:
        if i < current_index + WINDOW_SIZE:
            continue
        else:
            burst_centers.append(i)
            current_index = i

    return burst_centers

if __name__ == "__main__":
    spec = np.load("250106-Marquette-Senior-Hig.npy")
    times = np.load("250106-Marquette-Senior-Hig.npz")  # timestamps or seconds array

    burst_centers = get_high_values(spec.copy())

    show_burst_windows(
        "250106-Marquette-Senior-Hig",
        burst_centers,
        output_dir=DATA_DIR
    )