import os
import glob
import cv2
import numpy as np
import pandas as pd
import time
import torch
import pathlib

# --- CONFIGURATION ---
MODEL_PATH = './best.pt'
INPUT_FOLDER = 'test_dataset'
OUTPUT_FOLDER = 'detection_output'
SCORE_THRESHOLD = 0.5   # Minimum confidence
# ---------------------

# --- WINDOWS PATH FIX ---
# Necessary if the model was trained on Linux/Colab but running on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def draw_box(image, box, color, thickness=2):
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def get_color(label):
    np.random.seed(int(label))
    colors = np.random.randint(0, 255, size=(3,)).tolist()
    return tuple(colors)

def main():
    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load YOLOv5 Model (Clean Load)
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # This forces a download of the YOLOv5 code to cache, ignoring local conflicts
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        model.conf = SCORE_THRESHOLD 
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Create Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 4. Process Images
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*'))
    print(f"Found {len(image_paths)} images.")

    for img_path in image_paths:
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")

        # Read Image
        raw_image = cv2.imread(img_path)
        if raw_image is None:
            continue
        orig_height, orig_width, _ = raw_image.shape

        # --- INFERENCE ---
        start = time.time()
        results = model(img_path)
        print(f"  Time: {time.time() - start:.3f}s")

        # --- PARSING ---
        # Results are in results.xyxy[0] -> [x1, y1, x2, y2, confidence, class]
        detections = results.xyxy[0].cpu().numpy()

        # Prepare for CSV and Drawing
        csv_rows = []
        draw_img = raw_image.copy()

        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            score = det[4]
            cls = int(det[5])

            # Add to CSV list
            csv_rows.append({
                'image_name': filename,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': cls,
                'image_width': orig_width,
                'image_height': orig_height
            })

            # Draw
            color = get_color(cls)
            ##draw_box(draw_img, [x1, y1, x2, y2], color=(0, 255, 0))
            ##draw_caption(draw_img, [x1, y1, x2, y2], f"{score:.2f}")

        # --- SAVING ---
        # 1. Save CSV
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_path = os.path.join(OUTPUT_FOLDER, csv_filename)
        cols = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height']
        
        if csv_rows:
            df = pd.DataFrame(csv_rows, columns=cols)
        else:
            df = pd.DataFrame(columns=cols) # Empty CSV if no detections
            
        df.to_csv(csv_path, index=False)

        # 2. Save Image
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, draw_img)

    # Reset Pathlib (Clean up)
    pathlib.PosixPath = temp
    print(f"Done! Check folder: '{OUTPUT_FOLDER}'")

if __name__ == '__main__':
    main()