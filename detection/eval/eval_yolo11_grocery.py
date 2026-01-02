"""
YOLOv11 Inference & Evaluation Script for Grocery Products Dataset
Evaluates detection performance against CSV annotations and generates example outputs.
"""

import os
import cv2
import glob
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random

# --- YOLOv11 IMPORT ---
from ultralytics import YOLO

# --- CONFIGURATION ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, '../weights/weights_11S_new.pt')
DATASET_PATH = os.path.join(SCRIPT_DIR, '../../datasets/Grocery_products')
ANNOTATIONS_FOLDER = os.path.join(DATASET_PATH, 'Annotations')
TESTING_FOLDER = os.path.join(DATASET_PATH, 'Testing')
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, 'grocery_results')

CONF_THRESHOLD = 0.25
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
NUM_EXAMPLE_IMAGES = 6

# CPU OPTIMIZATIONS
BATCH_SIZE = 4
NUM_THREADS = 4
# ---------------------

# --- WINDOWS PATH FIX ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


def calculate_iou_batch(box_a, boxes_b):
    """Calculate IoU between one box and multiple boxes."""
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
    """Compute Average Precision from recall and precision arrays."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_csv_filename(csv_name):
    """
    Parse CSV filename to get store folder and image name.
    Format: s{store}_{image}.csv -> store{store}/images/{image}.jpg
    """
    base = os.path.splitext(csv_name)[0]
    parts = base.split('_')
    if len(parts) < 2:
        return None, None
    
    store_num = parts[0].replace('s', '')
    image_num = '_'.join(parts[1:])  # Handle cases like s1_14
    
    store_folder = f"store{store_num}"
    image_name = f"{image_num}.jpg"
    
    return store_folder, image_name


def load_ground_truth():
    """
    Load all ground truth annotations from CSV files.
    Returns: dict mapping image_path -> list of [x1, y1, x2, y2] boxes
    """
    gt_dict = {}
    image_paths = []
    
    csv_files = glob.glob(os.path.join(ANNOTATIONS_FOLDER, '*.csv'))
    print(f"Found {len(csv_files)} annotation CSV files.")
    
    for csv_path in csv_files:
        csv_name = os.path.basename(csv_path)
        store_folder, image_name = parse_csv_filename(csv_name)
        
        if store_folder is None:
            continue
        
        image_path = os.path.join(TESTING_FOLDER, store_folder, 'images', image_name)
        
        if not os.path.exists(image_path):
            continue
        
        # Read CSV annotations
        boxes = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 5:
                        # Format: path,x1,y1,x2,y2
                        try:
                            x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            # Handle negative coordinates (clamp to 0)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = max(0, x2), max(0, y2)
                            boxes.append([x1, y1, x2, y2])
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
        
        if boxes:
            gt_dict[image_path] = boxes
            image_paths.append(image_path)
    
    print(f"Loaded {len(image_paths)} images with {sum(len(v) for v in gt_dict.values())} total ground truth boxes.")
    return gt_dict, image_paths


def load_image(path):
    """Load image and convert to RGB."""
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def get_predictions_yolov11(model, image_paths):
    """Run YOLOv11 inference on all images."""
    preds_dict = {}
    total_files = len(image_paths)
    
    print(f"Starting YOLOv11 Inference on {total_files} images...")
    
    for i in tqdm(range(0, total_files, BATCH_SIZE)):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        
        # Threaded Load
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            batch_images = list(executor.map(load_image, batch_paths))
        
        valid_imgs = []
        valid_paths = []
        for img, path in zip(batch_images, batch_paths):
            if img is not None:
                valid_imgs.append(img)
                valid_paths.append(path)
            else:
                preds_dict[path] = np.empty((0, 5))
        
        if not valid_imgs:
            continue
        
        # YOLOv11 Batch Inference
        try:
            results = model(valid_imgs, conf=CONF_THRESHOLD, verbose=False)
            
            for j, result in enumerate(results):
                path = valid_paths[j]
                boxes = result.boxes.data.cpu().numpy()
                
                if len(boxes) > 0:
                    # [x1, y1, x2, y2, conf, cls] -> [x1, y1, x2, y2, conf]
                    preds_dict[path] = boxes[:, :5]
                else:
                    preds_dict[path] = np.empty((0, 5))
        
        except Exception as e:
            print(f"Inference error: {e}")
            for path in valid_paths:
                preds_dict[path] = np.empty((0, 5))
    
    return preds_dict


def evaluate(gt_dict, preds_dict, iou_thresh=0.5):
    """Evaluate predictions against ground truth."""
    true_positives = []
    scores = []
    
    num_gt_boxes = sum(len(gt_dict.get(path, [])) for path in preds_dict.keys())
    ious_list = []
    
    for path, pred_boxes in preds_dict.items():
        if len(pred_boxes) == 0:
            continue
        
        gt_boxes = np.array(gt_dict.get(path, []))
        
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


def draw_boxes_on_image(image_path, gt_boxes, pred_boxes):
    """Draw ground truth (red) and prediction (green) boxes on image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw ground truth boxes (red) - thicker line
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    # Draw prediction boxes (green) with confidence labels
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4] if len(box) > 4 else 0
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw confidence label with background for visibility
        label = f'{conf:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        label_y = max(y1 - 10, text_height + 5)
        cv2.rectangle(img, (x1, label_y - text_height - 5), 
                      (x1 + text_width + 5, label_y + 5), (0, 255, 0), -1)
        
        # Draw text (black on green background)
        cv2.putText(img, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), thickness)
    
    return img


def save_example_images(gt_dict, preds_dict, output_folder, num_examples=6):
    """Save example images with both GT and predictions."""
    # Select random images that have both GT and predictions
    valid_paths = [p for p in preds_dict.keys() 
                   if p in gt_dict and len(preds_dict[p]) > 0]
    
    if len(valid_paths) < num_examples:
        selected = valid_paths
    else:
        selected = random.sample(valid_paths, num_examples)
    
    print(f"Saving {len(selected)} example images...")
    
    for i, path in enumerate(selected):
        gt_boxes = gt_dict.get(path, [])
        pred_boxes = preds_dict.get(path, np.empty((0, 5)))
        
        img = draw_boxes_on_image(path, gt_boxes, pred_boxes)
        if img is not None:
            output_path = os.path.join(output_folder, f'example_{i+1}.png')
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {output_path}")


def plot_metrics(prec_50, rec_50, ap_50, ap_50_95, mean_iou, total_gt, total_pred, output_folder):
    """Generate and save metrics visualization."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('YOLOv11 Evaluation on Grocery Products Dataset', fontsize=16)

    # Precision-Recall Curve
    axs[0, 0].plot(rec_50, prec_50, color='blue', lw=2, label=f'AP@50 = {ap_50:.3f}')
    axs[0, 0].set_xlabel('Recall')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].set_title('Precision-Recall Curve (IoU=0.50)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_xlim(0, 1)
    axs[0, 0].set_ylim(0, 1)

    # Metric Summary Bars
    metrics = ['mAP@50', 'mAP@50-95', 'Mean IoU']
    values = [ap_50, ap_50_95, mean_iou]
    bars = axs[0, 1].bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axs[0, 1].set_ylim(0, 1.1)
    axs[0, 1].set_title('Metric Summary')
    for bar in bars:
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{bar.get_height():.3f}", ha='center', fontsize=12)

    # Detection Counts
    counts = ['Ground Truth', 'Predictions']
    count_values = [total_gt, total_pred]
    axs[1, 0].bar(counts, count_values, color=['#e74c3c', '#2ecc71'])
    axs[1, 0].set_title('Detection Counts')
    for i, v in enumerate(count_values):
        axs[1, 0].text(i, v + 10, str(v), ha='center', fontsize=12)

    # Text Summary
    axs[1, 1].axis('off')
    text_str = (f"Evaluation Summary\n"
                f"{'='*30}\n\n"
                f"mAP @ 50:      {ap_50:.4f}\n"
                f"mAP @ 50:95:   {ap_50_95:.4f}\n"
                f"Average IoU:   {mean_iou:.4f}\n\n"
                f"Ground Truth:  {total_gt} boxes\n"
                f"Predictions:   {total_pred} boxes\n"
                f"Confidence:    {CONF_THRESHOLD}")
    axs[1, 1].text(0.1, 0.5, text_str, fontsize=12, family='monospace',
                   bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 10},
                   verticalalignment='center')

    plt.tight_layout()
    output_path = os.path.join(output_folder, 'metrics_summary.png')
    plt.savefig(output_path, dpi=150)
    print(f"Metrics plot saved to: {output_path}")
    plt.close()


def main():
    print("=" * 50)
    print("YOLOv11 Grocery Products Evaluation")
    print("=" * 50)
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Load Model
    print(f"\nLoading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Load Ground Truth
    print("\nLoading ground truth annotations...")
    gt_dict, image_paths = load_ground_truth()
    
    if not image_paths:
        print("No valid images found!")
        return
    
    # 3. Run Predictions
    print("\nRunning inference...")
    preds_dict = get_predictions_yolov11(model, image_paths)
    
    # 4. Calculate Metrics
    print("\nCalculating metrics...")
    p50, r50, ap50, iou50 = evaluate(gt_dict, preds_dict, iou_thresh=0.5)
    
    ap_accum = []
    for thresh in IOU_THRESHOLDS:
        _, _, ap_val, _ = evaluate(gt_dict, preds_dict, iou_thresh=thresh)
        ap_accum.append(ap_val)
    
    map_50_95 = np.mean(ap_accum)
    
    # Count totals
    total_gt = sum(len(v) for v in gt_dict.values())
    total_pred = sum(len(v) for v in preds_dict.values())
    
    # 5. Print Results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"mAP@50:      {ap50:.4f}")
    print(f"mAP@50:95:   {map_50_95:.4f}")
    print(f"Mean IoU:    {iou50:.4f}")
    print(f"Total GT:    {total_gt} boxes")
    print(f"Total Pred:  {total_pred} boxes")
    print("=" * 50)
    
    # 6. Save Example Images
    print("\nGenerating example images...")
    save_example_images(gt_dict, preds_dict, OUTPUT_FOLDER, NUM_EXAMPLE_IMAGES)
    
    # 7. Plot Metrics
    print("\nGenerating metrics plot...")
    plot_metrics(p50, r50, ap50, map_50_95, iou50, total_gt, total_pred, OUTPUT_FOLDER)
    
    # Cleanup
    pathlib.PosixPath = temp
    
    print(f"\nDone! Check folder: '{OUTPUT_FOLDER}'")


if __name__ == '__main__':
    main()
