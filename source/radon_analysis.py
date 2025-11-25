import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.transform import radon
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import config

def radon_analysis():
    print("*\tStarting Enhanced Radon Transform Analysis...\n")

    # === Load the spectrogram CSV and NPY ===
    input_dir = config.INPUT_DIR_RA
    output_dir = config.OUTPUT_DIR_RA

    # Get all csv and npy files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    # Convert the csv files to npy
    numpy_data = []
    data_names = []
    if len(csv_files) > 0:
        for file_path in csv_files:
            data = np.loadtxt(file_path, delimiter=',')
            numpy_data.append(data)
            data_names.append(os.path.basename(file_path))

    # Load in numpy
    if len(npy_files) > 0:
        for file_path in npy_files:
            data = np.load(file_path)
            if len(data.shape) > 2:
                data = data[0]
            numpy_data.append(data)
            data_names.append(os.path.basename(file_path))

    total_files = len(npy_files) + len(csv_files)

    # Statistics
    num_detected = 0
    max_angle = -1.0
    min_angle = 360.0
    max_bursts = 0
    max_file = ""


    print(f"*\tLoaded in {total_files} files...")
    print(f" \t\t- {len(npy_files)} NPY files")
    print(f" \t\t- {len(csv_files)} CSV files\n")

    print(f"*\tBeginning Radon Analysis")
    for run in range(0, total_files):
        spectrogram = numpy_data[run]
        name = data_names[run]
        print(f" \tAnalyzing Spectrogram {name} Data:")
        print(f" \t\tSpectrogram shape: {spectrogram.shape}")
        print(f" \t\tAny NaNs? {np.isnan(spectrogram).any()}")

        # === Normalize spectrogram ===
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

        # === ENHANCEMENT 1: Adaptive thresholding to isolate bursts ===
        # This removes background noise and makes bursts stand out
        threshold = np.percentile(spectrogram, 75)  # Keep only top 25% of values
        spectrogram_thresh = np.where(spectrogram > threshold, spectrogram, 0)

        # === ENHANCEMENT 2: Contrast enhancement ===
        # Apply power transformation to increase contrast
        spectrogram_enhanced = np.power(spectrogram_thresh, 0.5)  # Square root for enhancement

        # === Save enhanced spectrogram ===
        if config.CREATE_IMG_RA:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].imshow(spectrogram, cmap='hot', aspect='auto')
            axes[0].set_title("Original Spectrogram")
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("Frequency")

            axes[1].imshow(spectrogram_thresh, cmap='hot', aspect='auto')
            axes[1].set_title("After Thresholding (Bursts Isolated)")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Frequency")

            axes[2].imshow(spectrogram_enhanced, cmap='hot', aspect='auto')
            axes[2].set_title("Enhanced (Ready for Radon)")
            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(f'{output_dir}/{name}_prepro_steps.png', dpi=300)
            plt.close()
        #print(" \tPreprocessing visualization...")

        # === ENHANCEMENT 3: More aggressive downsampling for speed ===
        # You can adjust this factor (higher = faster but less precise)
        downsample_factor = 4
        spectrogram_ds = spectrogram_enhanced[::downsample_factor, ::downsample_factor]
        #print(f" \tDownsampled spectrogram shape: {spectrogram_ds.shape}")

        # === ENHANCEMENT 4: Focused angle range ===
        # Most bursts appear at certain angles - we'll search a focused range first
        # Then expand if needed
        theta_focused = np.linspace(0., 180., 180, endpoint=False)  # 1 degree resolution

        #print("*\tComputing Radon transform with enhanced data...")
        radon_image = radon(spectrogram_ds, theta=theta_focused, circle=False)

        # === ENHANCEMENT 5: Better visualization with multiple views ===

        # 1. Original Radon (log scale)
        radon_log = np.log1p(radon_image)

        # 2. Normalized Radon (each column normalized separately)
        radon_normalized = radon_image.copy()
        for i in range(radon_normalized.shape[1]):
            col = radon_normalized[:, i]
            if col.max() > 0:
                radon_normalized[:, i] = (col - col.min()) / (col.max() - col.min())

        # 3. High-pass filtered to show peaks
        radon_smoothed = gaussian_filter(radon_image, sigma=3)
        radon_highpass = radon_image - radon_smoothed
        radon_highpass = np.clip(radon_highpass, 0, None)  # Keep only positive

        # === Create comprehensive visualization ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Log-scaled Radon
        if config.CREATE_IMG_RA:
            im1 = axes[0, 0].imshow(radon_log, cmap='magma', extent=(0, 180, 0, radon_log.shape[0]),
                                    aspect='auto', interpolation='bilinear')
            axes[0, 0].set_title("Radon Transform (Log-scaled)", fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel("Angle (degrees)")
            axes[0, 0].set_ylabel("Projection Position")
            plt.colorbar(im1, ax=axes[0, 0], label='Log(Intensity)')

            # Plot 2: Normalized Radon (best for seeing all angles clearly)
            im2 = axes[0, 1].imshow(radon_normalized, cmap='hot', extent=(0, 180, 0, radon_normalized.shape[0]),
                                    aspect='auto', interpolation='bilinear', vmin=0, vmax=1)
            axes[0, 1].set_title("Radon Transform (Normalized - BEST VIEW)", fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel("Angle (degrees)")
            axes[0, 1].set_ylabel("Projection Position")
            plt.colorbar(im2, ax=axes[0, 1], label='Normalized Intensity')

            # Plot 3: High-pass filtered (shows burst locations)
            im3 = axes[1, 0].imshow(radon_highpass, cmap='viridis', extent=(0, 180, 0, radon_highpass.shape[0]),
                                    aspect='auto', interpolation='bilinear')
            axes[1, 0].set_title("High-Pass Filtered (Burst Locations)", fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel("Angle (degrees)")
            axes[1, 0].set_ylabel("Projection Position")
            plt.colorbar(im3, ax=axes[1, 0], label='Filtered Intensity')

            # Plot 4: Intensity profile with peaks
            intensity_profile = np.mean(radon_image, axis=0)
            intensity_profile_smooth = gaussian_filter(intensity_profile, sigma=2)

            # Detect peaks with adaptive threshold
            peak_threshold = 0.3 * np.max(intensity_profile_smooth)
            peaks, properties = find_peaks(intensity_profile_smooth, height=peak_threshold, distance=5)
            dominant_angles = theta_focused[peaks]

            axes[1, 1].plot(theta_focused, intensity_profile, 'b-', alpha=0.3, label='Raw intensity')
            axes[1, 1].plot(theta_focused, intensity_profile_smooth, 'b-', linewidth=2, label='Smoothed intensity')
            axes[1, 1].plot(theta_focused[peaks], intensity_profile_smooth[peaks], "r*",
                            markersize=15, label=f'Detected peaks ({len(peaks)})')
            axes[1, 1].axhline(y=peak_threshold, color='g', linestyle='--', alpha=0.5, label='Detection threshold')
            axes[1, 1].set_title("Dominant Line Orientations", fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel("Angle (degrees)")
            axes[1, 1].set_ylabel("Average Intensity")
            axes[1, 1].legend()
            axes[1, 1].grid(True, linestyle='--', alpha=0.4)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/{name}_radon_analysis_enhanced.png', dpi=400)
            plt.close()
        #print("*\tEnhanced Radon analysis saved as 'radon_analysis_enhanced.png'")

        # === Print detected angles with interpretation ===
        #print("\n" + "="*60)
        #print("DETECTED BURST ORIENTATIONS:")
        #print("="*60)

        angle_text = ""
        if len(dominant_angles) > 0:
            for i, angle in enumerate(dominant_angles, 1):
                # Convert angle to slope interpretation
                if angle < 5 or angle > 175:
                    orientation = "Nearly horizontal (time-domain burst)"
                elif 85 < angle < 95:
                    orientation = "Nearly vertical (frequency sweep)"
                elif angle < 45:
                    orientation = "Shallow positive slope"
                elif angle < 90:
                    orientation = "Steep positive slope"
                elif angle < 135:
                    orientation = "Steep negative slope"
                else:
                    orientation = "Shallow negative slope"

                #print(f"Peak {i}: {angle:.1f}° - {orientation}")
        else:
            #print("No dominant angles detected. Bursts may be very diffuse or weak.")
            pass

        #print("="*60)

        # === Create a zoomed-in view of the most prominent angle ===
        num_peaks = len(peaks)
        if num_peaks > 0:
            # Find the strongest peak
            strongest_peak_idx = peaks[np.argmax(properties['peak_heights'])]
            strongest_angle = theta_focused[strongest_peak_idx]

            # Create zoomed view with better layout
            if config.CREATE_IMG_RA:
                fig = plt.figure(figsize=(18, 8))
                gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)

                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, :])

                # Full view with marker
                im1 = ax1.imshow(radon_normalized, cmap='hot', extent=(0, 180, 0, radon_normalized.shape[0]),
                                 aspect='auto', interpolation='bilinear')
                ax1.axvline(x=strongest_angle, color='cyan', linewidth=3, linestyle='--',
                            label=f'Strongest: {strongest_angle:.1f}°')
                ax1.set_title("Full Radon Transform", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Angle (degrees)", fontsize=12)
                ax1.set_ylabel("Projection Position", fontsize=12)
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                plt.colorbar(im1, ax=ax1, label='Normalized Intensity')

                # Zoomed slice at strongest angle - ENHANCED
                angle_idx = strongest_peak_idx
                window = 15  # degrees on each side (wider for more context)
                angle_start = max(0, angle_idx - window)
                angle_end = min(len(theta_focused), angle_idx + window)

                radon_slice = radon_normalized[:, angle_start:angle_end]

                # Apply additional contrast enhancement to the slice
                radon_slice_enhanced = np.power(radon_slice, 0.7)  # Power transform for better visibility

                im2 = ax2.imshow(radon_slice_enhanced, cmap='hot',  # High contrast colormap
                                 extent=(theta_focused[angle_start], theta_focused[angle_end-1], 0, radon_slice.shape[0]),
                                 aspect='auto', interpolation='bicubic', vmin=0, vmax=1)
                ax2.axvline(x=strongest_angle, color='cyan', linewidth=3, linestyle='--')
                ax2.set_title(f"Zoomed & Enhanced: ±{window}° around {strongest_angle:.1f}°",
                              fontsize=14, fontweight='bold')
                ax2.set_xlabel("Angle (degrees)", fontsize=12)
                ax2.set_ylabel("Projection Position (WHERE burst occurs)", fontsize=12)
                ax2.grid(True, alpha=0.3)
                plt.colorbar(im2, ax=ax2, label='Enhanced Intensity')

            # Add a 1D profile plot showing intensity along the strongest angle
            strongest_profile = radon_image[:, strongest_peak_idx]
            strongest_profile_norm = (strongest_profile - strongest_profile.min()) / (strongest_profile.max() - strongest_profile.min())

            # Find peaks in this profile to show burst locations
            profile_peaks, _ = find_peaks(strongest_profile_norm, height=0.4, distance=20)

            if config.CREATE_IMG_RA:
                ax3.plot(range(len(strongest_profile_norm)), strongest_profile_norm, 'b-', linewidth=2,
                         label=f'Intensity profile at {strongest_angle:.1f}°')
                ax3.plot(profile_peaks, strongest_profile_norm[profile_peaks], 'ro', markersize=12,
                         markeredgecolor='darkred', markeredgewidth=2, label=f'Burst locations ({len(profile_peaks)})')
                ax3.fill_between(range(len(strongest_profile_norm)), 0, strongest_profile_norm,
                                 alpha=0.3, color='lightblue')
                ax3.set_title(f"Burst Intensity Profile at {strongest_angle:.1f}° (Shows EXACT burst positions)",
                              fontsize=14, fontweight='bold')
                ax3.set_xlabel("Position along projection (relates to time/frequency in original data)", fontsize=12)
                ax3.set_ylabel("Normalized Intensity", fontsize=12)
                ax3.legend(fontsize=12)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim([-0.05, 1.05])

                plt.tight_layout()
                plt.savefig(f'{output_dir}/{name}_burst_location_detail.png', dpi=400, bbox_inches='tight')
                plt.close()

            num_bursts = len(profile_peaks)
            min_angle = min(min_angle, strongest_angle)
            max_angle = max(max_angle, strongest_angle)

            num_detected += num_bursts
            if num_bursts > max_bursts:
                max_bursts = num_bursts
                max_file = name
            print(f" \t\tBursts found: {num_bursts}")
            print(f" \tRun {run+1}/{total_files} completed!\n")

    print(f"*\t{config.GREEN}All Analysis complete!{config.RESET}")
    print("*\tStatistics:")
    print(f" \tBursts found: {num_detected}")
    print(f" \tMax angle found: {max_angle}")
    print(f" \tMin angle found: {min_angle}")
    print(f" \tMax bursts: {max_bursts} from {max_file}")

if __name__ == "__main__":
    radon_analysis()