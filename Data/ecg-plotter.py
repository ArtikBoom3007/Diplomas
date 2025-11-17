import os
import numpy as np
from pyedflib import highlevel
from pathlib import Path
import ecg_plot
from loguru import logger

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ---------------------------------------------------------------
# 1. FILTER DESIGN HELPERS
# ---------------------------------------------------------------

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='high', analog=False)

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def apply_notch_filter(signal, fs, freq=50.0, Q=30):
    """Notch filter for 50/60 Hz powerline noise."""
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, signal)

# ---------------------------------------------------------------
# 2. MAIN FILTERING FUNCTION
# ---------------------------------------------------------------

def filter_ecg(ecg, fs):
    """
    Removes:
        - baseline drift
        - high-frequency noise
        - powerline noise (optional)
    """

    # --- Remove baseline wander (0.5 Hz high-pass) ---
    hp_cutoff = 0.5
    b, a = butter_highpass(hp_cutoff, fs)
    ecg_hp = filtfilt(b, a, ecg)

    # --- Remove high-frequency noise (40 Hz low-pass) ---
    lp_cutoff = 40
    b, a = butter_lowpass(lp_cutoff, fs)
    ecg_bp = filtfilt(b, a, ecg_hp)

    # --- Remove 50/60 Hz interference ---
    ecg_clean = apply_notch_filter(ecg_bp, fs, freq=50)  # change to 60 if needed

    return ecg_clean


# ---------------------------------------------------------------
# 3. NORMALIZATION FOR CLASSIFICATION
# ---------------------------------------------------------------

def preprocess_ecg(ecg):
    """
    Standard preprocessing:
        - z-score normalization
        - (optional) min-max normalization
    """
    # Z-score normalization
    ecg_norm = (ecg - np.mean(ecg)) / np.std(ecg)

    # Optionally (replace above with this):
    # ecg_norm = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))

    return ecg_norm


# ---------------------------------------------------------------
# 4. OPTIONAL: SEGMENT INTO FIXED-WIDTH WINDOWS
# ---------------------------------------------------------------

def segment_ecg(ecg, window_size, step=None):
    """
    Creates overlapping or non-overlapping windows.
    window_size in samples.
    """
    if step is None:
        step = window_size  # non-overlap

    segments = []
    for start in range(0, len(ecg) - window_size, step):
        segments.append(ecg[start:start + window_size])

    return np.array(segments)


# ---------------------------------------------------------------
# 5. EXAMPLE USAGE
# ---------------------------------------------------------------
# if __name__ == "__main__":
#     # Example: create synthetic ECG
#     fs = 250  # sampling frequency
#     t = np.linspace(0, 10, fs * 10)
#     raw_ecg = np.sin(2*np.pi*1.2*t) + 0.2*np.random.randn(len(t))

#     # Filter
#     filtered = filter_ecg(raw_ecg, fs)

#     # Normalize
#     normalized = preprocess_ecg(filtered)

#     # Segment into 2-second windows
#     segments = segment_ecg(normalized, window_size=2*fs)

#     print("Raw ECG shape:", raw_ecg.shape)
#     print("Filtered ECG shape:", filtered.shape)
#     print("Segments:", segments.shape)


def plot_ecg_from_edf(edf_path, output_folder, width=1920, height=1080):
    """
    Read EDF file and create ECG plot image.
    
    Args:
        edf_path: Path to the EDF file
        output_folder: Folder to save the output images
        width: Plot width in pixels
        height: Plot height in pixels
    """
    logger.info(f"Processing: {edf_path}")
    
    # Read EDF file
    signals, signal_headers, headers = highlevel.read_edf(edf_path)
    if signals is None:
        logger.info(f"Skipping {edf_path} due to read error")
        return
    signals = signals[:, :5000]

    filtered = filter_ecg(signals, 500)
    # Create output filename
    base_name = Path(edf_path).stem

    # Assuming 'signals' contains your ECG data
    ecg_plot.plot(filtered, title="ECG 6", lead_index=['ECG I', 'ECG II', 'ECG V1', 'ECG V2', 'ECG V2', 'ECG V3', 'ECG V4', 'ECG V5', 'ECG V6'], sample_rate=500) # Adjust sample_rate as needed
    ecg_plot.save_as_svg(Path(edf_path).name, output_folder + "/")
    logger.info("Saved image to: {}", output_folder + "/" + Path(edf_path).name)

def process_edf_folder(input_folder, output_folder="ecg_plots", 
                       width=1920, height=1080):
    """
    Process all EDF files in a folder and save ECG plots.
    
    Args:
        input_folder: Folder containing EDF files
        output_folder: Folder to save output images
        width: Plot width in pixels
        height: Plot height in pixels
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all EDF files
    edf_files = list(Path(input_folder).glob("*.edf"))
    edf_files.extend(list(Path(input_folder).glob("*.EDF")))
    
    if not edf_files:
        logger.info(f"No EDF files found in {input_folder}")
        return
    
    logger.info(f"Found {len(edf_files)} EDF files")
    logger.info(f"Output folder: {output_folder}\n")
    
    # Process each file
    for edf_file in edf_files:
        plot_ecg_from_edf(str(edf_file), output_folder, width, height)
    
    logger.info(f"\nProcessing complete! Images saved to: {output_folder}")

if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "/workspaces/Diplomas/Data/AMY_add/AMY/2"  # Change this to your folder path
    OUTPUT_FOLDER = "/workspaces/Diplomas/Data/images/experiment/AMY_add"  # Output folder for images
    PLOT_WIDTH = 1920
    PLOT_HEIGHT = 1080
    
    # Process all EDF files
    process_edf_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )