"""
Detection + Classification Demo Pipeline

This script combines:
1. YOLOv11 detection to find product bounding boxes
2. ArcFace classification to identify each detected product

Usage:
    python demo.py --input <photo_folder>
    python demo.py --input detection/inference/detection_output
"""

import os
import sys
import argparse
import time
import glob
import cv2
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- YOLOv11 IMPORT ---
from ultralytics import YOLO

# --- PLANOGRAM IMPORT ---
sys.path.insert(0, str(Path(__file__).parent))
from planogram.planogram import generate_planogram, detect_shelf_lines, assign_shelves, print_planogram_summary

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
DETECTION_WEIGHTS = SCRIPT_DIR / "detection" / "weights" / "weights_11S_new.pt"
CLASSIFICATION_CHECKPOINT = SCRIPT_DIR / "classification" / "checkpoints" / "best_2.pth"
REFERENCE_DB_PATH = SCRIPT_DIR / "classification" / "eval" / "outputs" / "reference_db.pt"
OUTPUT_FOLDER = SCRIPT_DIR / "demo_output"

# Detection settings
CONF_THRESHOLD = 0.25

# Classification settings
EMBEDDING_DIM = 512
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WINDOWS PATH FIX ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# =============================================================================
# ArcFace Model Architecture (from evaluate_arcface.py)
# =============================================================================
class ArcFaceHead(nn.Module):
    """ArcFace head for classification."""
    def __init__(self, in_features, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.current_margin = 0.0

    def set_margin(self, margin):
        self.current_margin = min(margin, self.margin)

    def forward(self, embeddings, labels):
        return embeddings  # Not needed for inference

    def get_proxies(self):
        return F.normalize(self.weight, dim=1)


class HALHead(nn.Module):
    """Hierarchical Auxiliary Loss head (placeholder for inference)."""
    def __init__(self, class_to_category, category_to_idx, scale=64.0):
        super().__init__()
        self.scale = scale
        self.num_categories = len(category_to_idx)

    def forward(self, embeddings, cat_labels, class_proxies, class_to_idx):
        return None


class ProductRecognitionModel(nn.Module):
    def __init__(self, num_classes, num_categories, class_to_category, category_to_idx,
                 class_to_idx, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()
        
        # Backbone: ResNet-34
        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # ArcFace head
        self.arcface = ArcFaceHead(embedding_dim, num_classes, scale, margin)
        
        # HAL head
        self.hal = HALHead(class_to_category, category_to_idx, scale)
        
        self.class_to_idx = class_to_idx
        self.embedding_dim = embedding_dim

    def get_embeddings(self, x):
        features = self.backbone(x).flatten(1)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, dim=1)

    def forward(self, x, class_labels=None, cat_labels=None, compute_hal=True):
        embeddings = self.get_embeddings(x)
        return embeddings


# =============================================================================
# Helper Functions
# =============================================================================
def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Generate a deterministic color for a class name."""
    # Use hash to get consistent colors
    hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:8], 16)
    np.random.seed(hash_val % (2**32))
    color = tuple(np.random.randint(50, 255, size=3).tolist())
    return color


def load_classification_model(checkpoint_path: Path, ref_db_path: Path):
    """Load the ArcFace classification model and reference database."""
    
    if not ref_db_path.exists():
        print(f"[ERROR] Reference database not found: {ref_db_path}")
        print("Please run classification/eval/evaluate_arcface.py first to build it.")
        return None, None, None, None
    
    # Load reference database
    print(f"Loading reference database from {ref_db_path}...")
    ref_data = torch.load(ref_db_path, weights_only=False, map_location=DEVICE)
    ref_embeddings = ref_data['embeddings']
    ref_class_names = ref_data['class_names']
    
    # Build class mappings from reference database
    unique_classes = list(set(ref_class_names))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Extract categories from class names
    class_to_category = {}
    categories = set()
    for cls in unique_classes:
        parts = cls.split('/')
        category = parts[0] if len(parts) > 1 else 'Unknown'
        class_to_category[cls] = category
        categories.add(category)
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}
    
    # Build model
    print(f"Loading classification model from {checkpoint_path}...")
    model = ProductRecognitionModel(
        num_classes=len(class_to_idx),
        num_categories=len(category_to_idx),
        class_to_category=class_to_category,
        category_to_idx=category_to_idx,
        class_to_idx=class_to_idx,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)
    
    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint: {e}")
        print("Continuing with reference database only...")
    
    return model, ref_embeddings, ref_class_names, idx_to_class


def classify_crop(model, crop_image: np.ndarray, ref_embeddings: np.ndarray, 
                  ref_class_names: List[str]) -> Tuple[str, float]:
    """Classify a single crop using embedding similarity."""
    
    # Transform for inference
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert BGR to RGB and to PIL
    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    
    # Get embedding
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        query_embedding = model.get_embeddings(input_tensor).cpu().numpy()
    
    # Compute similarities
    similarities = np.dot(query_embedding, ref_embeddings.T).flatten()
    
    # Get top-1 prediction
    top_idx = np.argmax(similarities)
    predicted_class = ref_class_names[top_idx]
    confidence = similarities[top_idx]
    
    # Simplify class name for display (just the product file name)
    display_name = Path(predicted_class).stem
    
    return display_name, confidence


def run_detection(model, image_path: str) -> Tuple[List[Dict], float]:
    """Run YOLOv11 detection on a single image. Returns detections and time taken."""
    
    start_time = time.time()
    results = model(image_path, conf=CONF_THRESHOLD, verbose=False)
    detection_time = time.time() - start_time
    
    detections = []
    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            detections.append({
                'x1': int(x1), 'y1': int(y1), 
                'x2': int(x2), 'y2': int(y2),
                'detection_conf': float(conf),
                'detection_class': int(cls)
            })
    
    return detections, detection_time


def process_image(
    image_path: str,
    detection_model,
    classification_model,
    ref_embeddings: np.ndarray,
    ref_class_names: List[str],
    output_folder: Path
) -> Tuple[pd.DataFrame, Dict]:
    """Process a single image: detect, classify, and save results. Returns DataFrame and timing info."""
    
    timing = {'detection': 0, 'classification': 0, 'total': 0}
    total_start = time.time()
    
    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARNING] Could not load image: {image_path}")
        return pd.DataFrame()
    
    img_h, img_w = img.shape[:2]
    
    # Run detection
    detections, det_time = run_detection(detection_model, image_path)
    timing['detection'] = det_time
    
    if not detections:
        print(f"  No detections in {filename}")
        return pd.DataFrame(), timing
    
    # Classify each detection
    results = []
    class_colors = {}
    
    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop
        crop = img[y1:y2, x1:x2]
        
        # Classify (with timing)
        cls_start = time.time()
        predicted_class, class_conf = classify_crop(
            classification_model, crop, ref_embeddings, ref_class_names
        )
        timing['classification'] += time.time() - cls_start
        
        # Store result
        results.append({
            'image_name': filename,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'detection_conf': det['detection_conf'],
            'predicted_class': predicted_class,
            'class_confidence': class_conf
        })
        
        # Get/cache color for class
        if predicted_class not in class_colors:
            class_colors[predicted_class] = get_color_for_class(predicted_class)
    
    if not results:
        return pd.DataFrame(), timing
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = output_folder / f"{base_name}_results.csv"
    df.to_csv(csv_path, index=False)
    
    # Draw annotated image
    annotated_img = img.copy()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        pred_class = row['predicted_class']
        conf = row['class_confidence']
        
        color = class_colors[pred_class]
        
        # Draw box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{pred_class[:15]} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        label_y = max(y1 - 5, text_h + 5)
        cv2.rectangle(annotated_img, (x1, label_y - text_h - 5), 
                      (x1 + text_w + 5, label_y + 5), color, -1)
        cv2.putText(annotated_img, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), thickness)
    
    # Save annotated image
    annotated_path = output_folder / f"{base_name}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated_img)
    
    timing['total'] = time.time() - total_start
    
    print(f"  {filename}: {len(results)} products | Det: {timing['detection']:.2f}s | Cls: {timing['classification']:.2f}s | Total: {timing['total']:.2f}s")
    
    return df, timing


def main():
    parser = argparse.ArgumentParser(description='Detection + Classification Demo')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input folder containing images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder (default: demo_output)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--planogram', action='store_true',
                        help='Generate planogram after processing each image')
    parser.add_argument('--show-images', action='store_true',
                        help='Render product crops inside planogram cells (requires --planogram)')
    args = parser.parse_args()
    
    conf_threshold = args.conf
    
    input_folder = Path(args.input)
    output_folder = Path(args.output) if args.output else OUTPUT_FOLDER
    
    if not input_folder.exists():
        print(f"[ERROR] Input folder not found: {input_folder}")
        return
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Detection + Classification Demo")
    print("=" * 60)
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Device: {DEVICE}")
    print()
    
    # 1. Load Detection Model
    print("Loading YOLOv11 detection model...")
    if not DETECTION_WEIGHTS.exists():
        print(f"[ERROR] Detection weights not found: {DETECTION_WEIGHTS}")
        return
    detection_model = YOLO(str(DETECTION_WEIGHTS))
    
    # 2. Load Classification Model
    print("\nLoading ArcFace classification model...")
    classification_model, ref_embeddings, ref_class_names, idx_to_class = \
        load_classification_model(CLASSIFICATION_CHECKPOINT, REFERENCE_DB_PATH)
    
    if classification_model is None:
        return
    
    print(f"Reference database: {len(ref_class_names)} embeddings")
    
    # 3. Find Images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(input_folder / ext)))
    
    if not image_paths:
        print(f"\n[ERROR] No images found in {input_folder}")
        return
    
    print(f"\nFound {len(image_paths)} images to process")
    print("-" * 60)
    
    # 4. Process Images
    all_results = []
    all_timings = []
    total_start = time.time()
    
    for img_path in tqdm(image_paths, desc="Processing"):
        df, timing = process_image(
            img_path,
            detection_model,
            classification_model,
            ref_embeddings,
            ref_class_names,
            output_folder
        )
        all_timings.append(timing)
        if not df.empty:
            all_results.append(df)

            # Generate planogram for this image if requested
            if args.planogram:
                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2] if img is not None else (0, 0)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                plano_path = str(output_folder / f"{base_name}_planogram.png")
                print(f"  Generating planogram...")
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig = generate_planogram(
                    df, img_h, img_w,
                    image_path=img_path,
                    output_path=plano_path,
                    show_images=args.show_images,
                    title=f"Planogram — {base_name}",
                )
                plt.close(fig)
    
    total_pipeline_time = time.time() - total_start
    
    # 5. Save Combined Results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = output_folder / "all_results.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to: {combined_path}")
        
        # Print class distribution
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)
        class_counts = combined_df['predicted_class'].value_counts()
        for cls, count in class_counts.head(10).items():
            print(f"  {cls}: {count}")
        if len(class_counts) > 10:
            print(f"  ... and {len(class_counts) - 10} more classes")
        
        # Print timing summary
        total_det = sum(t['detection'] for t in all_timings)
        total_cls = sum(t['classification'] for t in all_timings)
        avg_det = total_det / len(all_timings) if all_timings else 0
        avg_cls = total_cls / len(all_timings) if all_timings else 0
        
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        print(f"  Total pipeline time:     {total_pipeline_time:.2f}s")
        print(f"  Total detection time:    {total_det:.2f}s")
        print(f"  Total classification:    {total_cls:.2f}s")
        print(f"  Avg detection/image:     {avg_det:.3f}s")
        print(f"  Avg classification/img:  {avg_cls:.3f}s")
        print(f"  Images processed:        {len(image_paths)}")
        print(f"  Throughput:              {len(image_paths)/total_pipeline_time:.2f} img/s")
    
    # Cleanup
    pathlib.PosixPath = temp
    
    print(f"\nDone! Check folder: '{output_folder}'")


if __name__ == '__main__':
    main()
