import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_npy(filename: str, save: bool = False):
    """
    Plot frequency vs time spectrogram from npy file
    """
    base = Path(filename).with_suffix("")
    npy_path = base.with_suffix(".npy")
    meta_path = base.with_suffix(".npz")
    values = np.load(npy_path)
    print(f"✅ Loaded data (shape={values.shape})")

    # Load metadata if available
    times = None
    freqs = None
    if meta_path and os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        times = meta.get("times")
        freqs = meta.get("frequencies")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[
            0,
            len(times) if times is not None else values.shape[1],
            freqs.min() if freqs is not None else 0,
            freqs.max() if freqs is not None else values.shape[0],
        ],
    )
    plt.colorbar(label="Amplitude / Value")

    # X-axis (Time)
    if times is not None:
        num_ticks = min(6, len(times))
        tick_indices = np.linspace(0, len(times) - 1, num_ticks, dtype=int)
        time_labels = [np.datetime_as_string(t, unit="s").split("T")[-1] for t in times[tick_indices]]
        plt.xticks(tick_indices, time_labels, rotation=30, ha="right")
        plt.xlabel("Time (UTC)")
    else:
        plt.xlabel("Time Index")

    # Y-axis (Frequency)
    if freqs is not None:
        plt.ylabel("Frequency (Hz)")
    else:
        plt.ylabel("Frequency Index")

    plt.title("Frequency-Time Plot")
    plt.tight_layout()
    if save:
        plt.savefig(base.with_suffix(".png"))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to NPY/NPZ for time-frequency data.")
    parser.add_argument("filename", type=str, help="Path to CSV file")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save the graph as a png")
    args = parser.parse_args()

    plot_npy(args.filename, save=args.save)


if __name__ == "__main__":
    main()


