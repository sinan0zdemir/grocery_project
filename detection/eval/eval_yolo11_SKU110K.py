import os
import cv2
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- NEW IMPORT FOR YOLOv11 ---
from ultralytics import YOLO 

# --- CONFIGURATION ---
MODEL_PATH = '../weights/weights_11S_new.pt'     # Path to your best.pt (YOLOv11 trained)
IMAGE_FOLDER = '../../SKU110K/images'
GT_CSV_PATH = '../../SKU110K/annotations/annotations_test.csv' # Path to your Ground Truth CSV
CONF_THRESHOLD = 0.35 
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

# CPU OPTIMIZATIONS
BATCH_SIZE = 4
NUM_THREADS = 4
# ---------------------

# --- WINDOWS PATH FIX (Usually not needed for v11, but kept for safety) ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def calculate_iou_batch(box_a, boxes_b):
    x_a = np.maximum(box_a[0], boxes_b[:, 0])
    y_a = np.maximum(box_a[1], boxes_b[:, 1])
    x_b = np.minimum(box_a[2], boxes_b[:, 2])
    y_b = np.minimum(box_a[3], boxes_b[:, 3])

    inter_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxes_b_area = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union_area = box_a_area + boxes_b_area - inter_area
    return inter_area / (union_area + 1e-6)

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def load_ground_truth(csv_path):
    cols = ['filename', 'x1', 'y1', 'x2', 'y2', 'class', 'width', 'height']
    try:
        df = pd.read_csv(csv_path, header=None, names=cols, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}, []

    df['filename'] = df['filename'].astype(str).str.strip()
    unique_files = df['filename'].unique().tolist()
    gt_dict = {}
    for _, row in df.iterrows():
        fname = row['filename']
        try:
            box = [float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])]
            if fname not in gt_dict:
                gt_dict[fname] = []
            gt_dict[fname].append(box)
        except ValueError:
            continue
    return gt_dict, unique_files

def load_image(path):
    if os.path.exists(path):
        # OpenCV loads as BGR
        img = cv2.imread(path)
        if img is not None:
            # YOLOv11 prefers RGB, though it handles BGR. 
            # Converting to RGB explicitly is safer for consistency.
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def get_predictions_yolov11(model, image_folder, file_list):
    preds_dict = {}
    total_files = len(file_list)
    
    print(f"Starting YOLOv11 Inference on {total_files} images...")
    
    # Iterate in batches
    for i in tqdm(range(0, total_files, BATCH_SIZE)):
        batch_filenames = file_list[i : i + BATCH_SIZE]
        batch_paths = [os.path.join(image_folder, f) for f in batch_filenames]
        
        # 1. Threaded Load
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            batch_images = list(executor.map(load_image, batch_paths))

        valid_imgs = []
        valid_fnames = []
        for img, fname in zip(batch_images, batch_filenames):
            if img is not None:
                valid_imgs.append(img)
                valid_fnames.append(fname)
            else:
                preds_dict[fname] = np.empty((0, 5))

        if not valid_imgs:
            continue

        # 2. YOLOv11 Batch Inference
        try:
            # verbose=False keeps the console clean
            results = model(valid_imgs, conf=CONF_THRESHOLD, verbose=False)
            
            # 3. Parse YOLOv11 Results
            for j, result in enumerate(results):
                filename = valid_fnames[j]
                
                # result.boxes.data contains [x1, y1, x2, y2, conf, cls]
                # We need to move it to CPU and convert to numpy
                boxes = result.boxes.data.cpu().numpy()
                
                if len(boxes) > 0:
                    # We only need [x1, y1, x2, y2, conf] for evaluation
                    # boxes[:, :5] takes the first 5 columns
                    preds_dict[filename] = boxes[:, :5]
                else:
                    preds_dict[filename] = np.empty((0, 5))

        except Exception as e:
            print(f"Inference error: {e}")
            for fname in valid_fnames:
                preds_dict[fname] = np.empty((0, 5))
        
    return preds_dict

def evaluate(gt_dict, preds_dict, iou_thresh=0.5):
    true_positives = []
    scores = []
    
    num_gt_boxes = 0
    for fname in preds_dict.keys():
        if fname in gt_dict:
            num_gt_boxes += len(gt_dict[fname])

    ious_list = []

    for filename, pred_boxes in preds_dict.items():
        if len(pred_boxes) == 0:
            continue
            
        gt_boxes = np.array(gt_dict.get(filename, []))
        
        sorted_indices = np.argsort(-pred_boxes[:, 4])
        pred_boxes = pred_boxes[sorted_indices]
        
        detected_gt = []
        
        for p_box in pred_boxes:
            scores.append(p_box[4])
            if len(gt_boxes) == 0:
                true_positives.append(0)
                continue
            ious = calculate_iou_batch(p_box[:4], gt_boxes)
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]
            
            if best_iou >= iou_thresh and best_iou_idx not in detected_gt:
                true_positives.append(1)
                detected_gt.append(best_iou_idx)
                ious_list.append(best_iou)
            else:
                true_positives.append(0)

    true_positives = np.array(true_positives)
    scores = np.array(scores)
    
    if len(scores) == 0:
        return np.array([0]), np.array([0]), 0.0, 0.0

    indices = np.argsort(-scores)
    true_positives = true_positives[indices]
    
    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(1 - true_positives)
    
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (num_gt_boxes + 1e-6)
    ap = compute_ap(recall, precision)
    mean_iou = np.mean(ious_list) if ious_list else 0.0

    return precision, recall, ap, mean_iou

def plot_metrics(prec_50, rec_50, ap_50, ap_50_95, mean_iou):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'YOLOv11 Evaluation Results', fontsize=16)

    axs[0, 0].plot(rec_50, prec_50, color='blue', lw=2, label=f'AP@50 = {ap_50:.2f}')
    axs[0, 0].set_xlabel('Recall')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_title('Precision-Recall Curve (IoU=0.50)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    metrics = ['mAP@50', 'mAP@50-95', 'Mean IoU']
    values = [ap_50, ap_50_95, mean_iou]
    bars = axs[0, 1].bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axs[0, 1].set_ylim(0, 1.1)
    axs[0, 1].set_title('Metric Summary')
    for bar in bars:
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f"{bar.get_height():.3f}", ha='center')

    axs[1, 0].axis('off')
    text_str = (f"Summary:\n\n"
                f"mAP @ 50: {ap_50:.4f}\n"
                f"mAP @ 50:95: {ap_50_95:.4f}\n"
                f"Average IoU: {mean_iou:.4f}")
    axs[1, 0].text(0.1, 0.5, text_str, fontsize=12, bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 10})

    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig('evaluation_results11new.png')
    print("Results plot saved to evaluation_results11new.png")
    plt.show()

def main():
    print(f"Initializing YOLOv11...")
    
    # 1. Load Model (YOLOv11 Style)
    try:
        model = YOLO(MODEL_PATH) 
        # Note: We don't need .to(device) explicitly, Ultralytics handles it.
        # It will default to GPU if available, or CPU if not.
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load GT
    if not os.path.exists(GT_CSV_PATH):
        print(f"CSV not found: {GT_CSV_PATH}")
        return

    gt_dict, file_list = load_ground_truth(GT_CSV_PATH)
    
    if not file_list:
        print("No files found in CSV.")
        return

    # 3. Run Predictions
    preds_dict = get_predictions_yolov11(model, IMAGE_FOLDER, file_list)

    # 4. Calc Metrics
    print("\nCalculating Metrics...")
    p50, r50, ap50, iou50 = evaluate(gt_dict, preds_dict, iou_thresh=0.5)
    
    ap_accum = []
    for thresh in IOU_THRESHOLDS:
        _, _, ap_val, _ = evaluate(gt_dict, preds_dict, iou_thresh=thresh)
        ap_accum.append(ap_val)
    
    map_50_95 = np.mean(ap_accum)
    
    print(f"\nFinal Results:")
    print(f"mAP@50:    {ap50:.4f}")
    print(f"mAP@50:95: {map_50_95:.4f}")
    print(f"Mean IoU:  {iou50:.4f}")

    # 5. Plot
    plot_metrics(p50, r50, ap50, map_50_95, iou50)
    
    pathlib.PosixPath = temp

if __name__ == '__main__':
    main()