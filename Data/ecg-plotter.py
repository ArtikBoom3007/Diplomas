import os
import numpy as np
from pyedflib import highlevel
from pathlib import Path
import ecg_plot
from loguru import logger

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
    # Create output filename
    base_name = Path(edf_path).stem

    # Assuming 'signals' contains your ECG data
    ecg_plot.plot(signals, title="ECG 6", lead_index=['ECG I', 'ECG II', 'ECG V1', 'ECG V2', 'ECG V2', 'ECG V3', 'ECG V4', 'ECG V5', 'ECG V6'], sample_rate=500) # Adjust sample_rate as needed
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
    OUTPUT_FOLDER = "/workspaces/Diplomas/Data/images/AMY_add"  # Output folder for images
    PLOT_WIDTH = 1920
    PLOT_HEIGHT = 1080
    
    # Process all EDF files
    process_edf_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )