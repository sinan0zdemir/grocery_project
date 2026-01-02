"""
YOLOv11 Detection Evaluation Script for SDP_Product Dataset

Evaluates detection performance against Roboflow CSV annotations and generates example outputs.

Dataset format: Roboflow TensorFlow Object Detection CSV
    filename,width,height,class,xmin,ymin,xmax,ymax

Usage:
    python eval_yolo11_SDP.py [--weights PATH] [--dataset PATH] [--conf FLOAT]
"""

import os
import sys
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
from pathlib import Path
from collections import defaultdict

# --- YOLOv11 IMPORT ---
from ultralytics import YOLO

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Default paths
DEFAULT_MODEL_PATH = SCRIPT_DIR.parent / "weights" / "weights_11S_new.pt"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "SDP_Product.v1"
DEFAULT_OUTPUT_FOLDER = SCRIPT_DIR / "sdp_results"

CONF_THRESHOLD = 0.25
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
NUM_EXAMPLE_IMAGES = 6

# CPU OPTIMIZATIONS
BATCH_SIZE = 4
NUM_THREADS = 4

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


def load_ground_truth_roboflow(dataset_path: Path):
    """
    Load ground truth from Roboflow TensorFlow Object Detection CSV format.
    
    CSV format: filename,width,height,class,xmin,ymin,xmax,ymax
    
    Returns: 
        gt_dict: dict mapping image_path -> list of [x1, y1, x2, y2, class_name]
        image_paths: list of image paths
        class_names: set of unique class names
    """
    gt_dict = {}
    image_paths = []
    class_names = set()
    
    # Find all splits (train, valid, test)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
        
        csv_path = split_path / "_annotations.csv"
        if not csv_path.exists():
            continue
        
        print(f"  Loading {split} annotations from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Expected columns: filename, width, height, class, xmin, ymin, xmax, ymax
            required_cols = ['filename', 'xmin', 'ymin', 'xmax', 'ymax']
            if not all(col in df.columns for col in required_cols):
                print(f"    Warning: Missing required columns in {csv_path}")
                print(f"    Found columns: {df.columns.tolist()}")
                continue
            
            # Group by filename
            for filename, group in df.groupby('filename'):
                image_path = str(split_path / filename)
                
                if not os.path.exists(image_path):
                    continue
                
                boxes = []
                for _, row in group.iterrows():
                    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                    cls = row.get('class', 'product')
                    
                    # Clamp negative coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(0, x2), max(0, y2)
                    
                    boxes.append([x1, y1, x2, y2, cls])
                    class_names.add(cls)
                
                if boxes:
                    gt_dict[image_path] = boxes
                    image_paths.append(image_path)
        
        except Exception as e:
            print(f"    Error reading {csv_path}: {e}")
            continue
    
    print(f"Loaded {len(image_paths)} images with {sum(len(v) for v in gt_dict.values())} total ground truth boxes.")
    print(f"Found {len(class_names)} classes: {sorted(class_names)}")
    
    return gt_dict, image_paths, class_names


def load_image(path):
    """Load image and convert to RGB."""
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def get_predictions_yolov11(model, image_paths, conf_thresh):
    """Run YOLOv11 inference on all images."""
    preds_dict = {}
    total_files = len(image_paths)
    
    print(f"Starting YOLOv11 Inference on {total_files} images...")
    
    for i in tqdm(range(0, total_files, BATCH_SIZE), desc="Inference"):
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
                preds_dict[path] = np.empty((0, 6))
        
        if not valid_imgs:
            continue
        
        # YOLOv11 Batch Inference
        try:
            results = model(valid_imgs, conf=conf_thresh, verbose=False)
            
            for j, result in enumerate(results):
                path = valid_paths[j]
                boxes = result.boxes.data.cpu().numpy()
                
                if len(boxes) > 0:
                    # [x1, y1, x2, y2, conf, cls]
                    preds_dict[path] = boxes
                else:
                    preds_dict[path] = np.empty((0, 6))
        
        except Exception as e:
            print(f"Inference error: {e}")
            for path in valid_paths:
                preds_dict[path] = np.empty((0, 6))
    
    return preds_dict


def evaluate(gt_dict, preds_dict, iou_thresh=0.5):
    """Evaluate predictions against ground truth (class-agnostic for detection)."""
    true_positives = []
    scores = []
    
    num_gt_boxes = sum(len(gt_dict.get(path, [])) for path in preds_dict.keys())
    ious_list = []
    
    for path, pred_boxes in preds_dict.items():
        if len(pred_boxes) == 0:
            continue
        
        gt_boxes_raw = gt_dict.get(path, [])
        if not gt_boxes_raw:
            # All predictions are false positives
            for p_box in pred_boxes:
                scores.append(p_box[4])
                true_positives.append(0)
            continue
        
        gt_boxes = np.array([[b[0], b[1], b[2], b[3]] for b in gt_boxes_raw])
        
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
    
    # Draw ground truth boxes (red) with class labels
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls = box[4] if len(box) > 4 else ""
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Draw class label
        if cls:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(str(cls), font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), (255, 0, 0), -1)
            cv2.putText(img, str(cls), (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    # Draw prediction boxes (green) with confidence labels
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4] if len(box) > 4 else 0
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw confidence label
        label = f'{conf:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 10, text_height + 5)
        cv2.rectangle(img, (x1, label_y - text_height - 5), 
                      (x1 + text_width + 5, label_y + 5), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), thickness)
    
    return img


def save_example_images(gt_dict, preds_dict, output_folder, num_examples=6):
    """Save example images with both GT and predictions."""
    valid_paths = [p for p in preds_dict.keys() 
                   if p in gt_dict]
    
    if len(valid_paths) < num_examples:
        selected = valid_paths
    else:
        selected = random.sample(valid_paths, num_examples)
    
    print(f"Saving {len(selected)} example images...")
    
    for i, path in enumerate(selected):
        gt_boxes = gt_dict.get(path, [])
        pred_boxes = preds_dict.get(path, np.empty((0, 6)))
        
        img = draw_boxes_on_image(path, gt_boxes, pred_boxes)
        if img is not None:
            output_path = os.path.join(output_folder, f'example_{i+1}.png')
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {output_path}")


def plot_metrics(prec_50, rec_50, ap_50, ap_50_95, mean_iou, total_gt, total_pred, 
                 output_folder, conf_thresh):
    """Generate and save metrics visualization."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('YOLOv11 Evaluation on SDP_Product Dataset', fontsize=16)

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
        axs[1, 0].text(i, v + max(count_values)*0.02, str(v), ha='center', fontsize=12)

    # Text Summary
    axs[1, 1].axis('off')
    text_str = (f"Evaluation Summary\n"
                f"{'='*30}\n\n"
                f"mAP @ 50:      {ap_50:.4f}\n"
                f"mAP @ 50:95:   {ap_50_95:.4f}\n"
                f"Average IoU:   {mean_iou:.4f}\n\n"
                f"Ground Truth:  {total_gt} boxes\n"
                f"Predictions:   {total_pred} boxes\n"
                f"Confidence:    {conf_thresh}")
    axs[1, 1].text(0.1, 0.5, text_str, fontsize=12, family='monospace',
                   bbox={'facecolor': 'lightgrey', 'alpha': 0.5, 'pad': 10},
                   verticalalignment='center')

    plt.tight_layout()
    output_path = os.path.join(output_folder, 'metrics_summary.png')
    plt.savefig(output_path, dpi=150)
    print(f"Metrics plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 SDP_Product Detection Evaluation")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to YOLOv11 weights (.pt file)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH),
                        help="Path to SDP_Product dataset root")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FOLDER),
                        help="Output folder for results")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                        help="Confidence threshold for predictions")
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLE_IMAGES,
                        help="Number of example images to generate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv11 SDP_Product Detection Evaluation")
    print("=" * 60)
    
    model_path = Path(args.weights)
    dataset_path = Path(args.dataset)
    output_folder = Path(args.output)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    print(f"\n📦 Loading model from {model_path}...")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 2. Load Ground Truth
    print(f"\n📊 Loading ground truth from {dataset_path}...")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    gt_dict, image_paths, class_names = load_ground_truth_roboflow(dataset_path)
    
    if not image_paths:
        print("❌ No valid images found!")
        return
    
    # 3. Run Predictions
    print("\n🔍 Running inference...")
    preds_dict = get_predictions_yolov11(model, image_paths, args.conf)
    
    # 4. Calculate Metrics
    print("\n📈 Calculating metrics...")
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
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  mAP@50:      {ap50:.4f}")
    print(f"  mAP@50:95:   {map_50_95:.4f}")
    print(f"  Mean IoU:    {iou50:.4f}")
    print(f"  Total GT:    {total_gt} boxes")
    print(f"  Total Pred:  {total_pred} boxes")
    print("=" * 60)
    
    # 6. Save Results to file
    results_file = output_folder / "results.txt"
    with open(results_file, "w") as f:
        f.write("SDP_Product Detection Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Confidence Threshold: {args.conf}\n")
        f.write(f"Classes: {sorted(class_names)}\n\n")
        f.write("Metrics:\n")
        f.write(f"  mAP@50:      {ap50:.4f}\n")
        f.write(f"  mAP@50:95:   {map_50_95:.4f}\n")
        f.write(f"  Mean IoU:    {iou50:.4f}\n\n")
        f.write(f"Detection Counts:\n")
        f.write(f"  Ground Truth: {total_gt} boxes\n")
        f.write(f"  Predictions:  {total_pred} boxes\n")
    print(f"📄 Results saved to: {results_file}")
    
    # 7. Save Example Images
    print("\n🖼️ Generating example images...")
    save_example_images(gt_dict, preds_dict, str(output_folder), args.num_examples)
    
    # 8. Plot Metrics
    print("\n📊 Generating metrics plot...")
    plot_metrics(p50, r50, ap50, map_50_95, iou50, total_gt, total_pred, 
                 str(output_folder), args.conf)
    
    # Cleanup
    pathlib.PosixPath = temp
    
    print(f"\n🎉 Done! Check folder: '{output_folder}'")


if __name__ == '__main__':
    main()
