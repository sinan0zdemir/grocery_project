import os
import glob
import cv2
import pandas as pd
import numpy as np

# --- CONFIGURATION BASED ON YOUR TREE ---
# The folder containing t1.csv, t1.jpeg, etc.
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_DIR, 'detection_output')# The folder where crops will be saved
OUTPUT_FOLDER = 'products'
# ----------------------------------------

def main():
    # 1. Create Output Folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")

    # 2. Find CSV files in 'detection_output'
    # This looks for C:./detection_output/*.csv
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, '*.csv'))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{INPUT_FOLDER}'.")
        print("Double check that folder name matches exactly.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        print(f"Processing: {filename}")
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Error reading CSV: {e}")
            continue

        if df.empty:
            print("  CSV is empty, skipping.")
            continue

        # 3. Load the Image
        # We assume the image is in the SAME folder as the CSV ('detection_output')
        image_filename = df.iloc[0]['image_name']
        image_path = os.path.join(INPUT_FOLDER, image_filename)

        img = cv2.imread(image_path)
        
        if img is None:
            print(f"  [WARNING] Image file missing: {image_path}")
            continue
        
        img_h, img_w = img.shape[:2]

        # 4. Crop Products
        crop_count = 0
        for index, row in df.iterrows():
            crop_count += 1
            
            # Get coordinates
            x1 = int(row['x1'])
            y1 = int(row['y1'])
            x2 = int(row['x2'])
            y2 = int(row['y2'])

            # Safety checks (clamp to image size)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop
            crop_img = img[y1:y2, x1:x2]

            # Save
            base_name = os.path.splitext(image_filename)[0] # e.g., t1
            save_name = f"{base_name}_crop_{crop_count}.jpg"
            save_path = os.path.join(OUTPUT_FOLDER, save_name)
            
            cv2.imwrite(save_path, crop_img)

        print(f"  Saved {crop_count} crops.")

    print(f"\nSuccess! Check the '{OUTPUT_FOLDER}' folder.")

if __name__ == '__main__':
    main()