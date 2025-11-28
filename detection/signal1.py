import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks 
import cv2

# --- CONFIGURATION ---
OUTPUT_FOLDER = 'output_results_final'
PLOT_FOLDER = 'output_plots'
SMOOTHING_FACTOR = 0.15

# TUNING PARAMETERS
MIN_SHELF_SPACING = 50  
# Note: Since we are looking for minima (valleys), we don't use MIN_SIGNAL_STRENGTH 
# effectively anymore, as the "peak" of a valley is low density. 
# We rely on spacing and relative prominence.
# ---------------------

def calculate_vertical_signal(df, height):
    """Generates the vertical density signal."""
    if df.empty:
        return np.zeros(height), np.linspace(0, height, height)

    df['y_center'] = (df['y1'] + df['y2']) / 2
    y_positions = df['y_center'].values

    try:
        density = gaussian_kde(y_positions)
        density.covariance_factor = lambda: SMOOTHING_FACTOR
        density._compute_covariance()
        
        y_grid = np.linspace(0, height, height)
        signal = density(y_grid)
        
        # Normalize to 0-100
        signal = (signal / signal.max()) * 100
        return signal, y_grid
        
    except Exception:
        return np.zeros(height), np.linspace(0, height, height)

def main():
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)

    csv_files = glob.glob(os.path.join(OUTPUT_FOLDER, '*.csv'))
    print(f"Found {len(csv_files)} CSV files.")

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        print(f"Processing {filename}...")
        
        # 1. Load Data
        df = pd.read_csv(csv_path)
        if df.empty: continue

        img_h = int(df.iloc[0]['image_height'])
        img_w = int(df.iloc[0]['image_width'])
        image_name = df.iloc[0]['image_name']
        image_path = os.path.join(OUTPUT_FOLDER, image_name)
        
        # 2. Calculate Signal (Product Density)
        signal, y_grid = calculate_vertical_signal(df, img_h)

        # 3. FIND SHELF LINES (LOCAL MINIMA)
        # To find minima using find_peaks, we invert the signal.
        # High density becomes low, Low density (shelves) becomes high.
        inverted_signal = 100 - signal 
        
        # We use prominence to ensure we only pick deep valleys between products,
        # not just tiny noise in the background.
        peaks, _ = find_peaks(inverted_signal, distance=MIN_SHELF_SPACING, prominence=5)
        
        # Convert indices to Y-pixel coordinates
        shelf_y_positions = y_grid[peaks]

        # --- PLOTTING ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(f'Shelf Detection (Valleys): {image_name}', fontsize=14)

        # LEFT PLOT: Image with Shelf Lines
        img_bgr = cv2.imread(image_path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Draw the lines on the image
            for y in shelf_y_positions:
                y_int = int(y)
                # Draw Blue Line across the image
                cv2.line(img_rgb, (0, y_int), (img_w, y_int), (0, 0, 255), 3)
                # Add label
                cv2.putText(img_rgb, f"Y:{y_int}", (10, y_int - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            axes[0].imshow(img_rgb)
            axes[0].set_title(f"Detected Shelves: {len(peaks)}")
            axes[0].axis('off')

        # RIGHT PLOT: Signal with Markers
        axes[1].plot(signal, y_grid, color='blue', linewidth=2, label='Product Density')
        axes[1].fill_betweenx(y_grid, signal, 0, color='blue', alpha=0.3)
        
        # Mark the DETECTED MINIMA on the graph (plotted at the signal's low point)
        # We use the original signal values for plotting the "X" mark, not the inverted values
        axes[1].plot(signal[peaks], shelf_y_positions, "x", color='red', markersize=10, markeredgewidth=3, label='Shelf Line')
        
        # Draw corresponding lines on the graph
        for y in shelf_y_positions:
            axes[1].axhline(y=y, color='red', linestyle='--', alpha=0.5)

        # Setup Graph Axis
        axes[1].set_ylim(img_h, 0) # Invert Y to match image
        axes[1].set_xlim(0, 110)
        axes[1].set_title("Vertical Signal Analysis")
        axes[1].set_ylabel("Pixel Position (Y)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        
        plot_filename = os.path.splitext(filename)[0] + '_shelves.png'
        plt.savefig(os.path.join(PLOT_FOLDER, plot_filename))
        plt.close()

    print(f"Done! Check '{PLOT_FOLDER}' for images with lines.")

if __name__ == '__main__':
    main()