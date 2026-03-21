"""
Detection + Classification Demo Pipeline

This script combines:
1. YOLOv11 detection to find product bounding boxes
2. ArcFace classification to identify each detected product
3. Planogram generation from classification results

Usage:
    python demo.py --input <photo_folder>
    python demo.py --input detection/inference/detection_output

Output structure:
    demo_output/
        classification/   ← annotated images + per-image CSVs + all_results.csv
        detection/        ← detection-only annotated images
        planogram/        ← planogram images
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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- YOLOv11 IMPORT ---
from ultralytics import YOLO

# --- Planogram ---
sys.path.insert(0, str(Path(__file__).parent / "planogram"))
from planogram import generate_planogram

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
DETECTION_WEIGHTS       = SCRIPT_DIR / "detection" / "weights" / "weights_11S_new.pt"
CLASSIFICATION_CHECKPOINT = SCRIPT_DIR / "classification" / "checkpoints" / "best_2.pth"
REFERENCE_DB_PATH       = SCRIPT_DIR / "classification" / "eval" / "outputs" / "reference_db_migros.pt"
OUTPUT_FOLDER           = SCRIPT_DIR / "demo_output"

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

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
# ArcFace Model Architecture
# =============================================================================
class ArcFaceHead(nn.Module):
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
        return embeddings

    def get_proxies(self):
        return F.normalize(self.weight, dim=1)


class HALHead(nn.Module):
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
        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        self.arcface = ArcFaceHead(embedding_dim, num_classes, scale, margin)
        self.hal = HALHead(class_to_category, category_to_idx, scale)
        self.class_to_idx = class_to_idx
        self.embedding_dim = embedding_dim

    def get_embeddings(self, x):
        features = self.backbone(x).flatten(1)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, dim=1)

    def forward(self, x, class_labels=None, cat_labels=None, compute_hal=True):
        return self.get_embeddings(x)


# =============================================================================
# Helper Functions
# =============================================================================
def get_color_for_class(class_name: str) -> Tuple[int, int, int]:
    hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:8], 16)
    np.random.seed(hash_val % (2**32))
    color = tuple(np.random.randint(50, 255, size=3).tolist())
    return color


def load_classification_model(checkpoint_path: Path, ref_db_path: Path):
    if not ref_db_path.exists():
        print(f"[ERROR] Reference database not found: {ref_db_path}")
        return None, None, None, None

    print(f"Loading reference database from {ref_db_path}...")
    ref_data = torch.load(ref_db_path, weights_only=False, map_location=DEVICE)
    ref_embeddings = ref_data['embeddings']
    ref_class_names = ref_data['class_names']

    unique_classes = list(set(ref_class_names))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    class_to_category = {}
    categories = set()
    for cls in unique_classes:
        parts = cls.split('/')
        category = parts[0] if len(parts) > 1 else 'Unknown'
        class_to_category[cls] = category
        categories.add(category)
    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}

    print(f"Loading classification model from {checkpoint_path}...")
    model = ProductRecognitionModel(
        num_classes=len(class_to_idx),
        num_categories=len(category_to_idx),
        class_to_category=class_to_category,
        category_to_idx=category_to_idx,
        class_to_idx=class_to_idx,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        filtered = {k: v for k, v in state_dict.items() if k.startswith('backbone.') or k.startswith('embedding.')}
        model.load_state_dict(filtered, strict=False)
        model.eval()
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint: {e}")
        print("Continuing with reference database only...")

    return model, ref_embeddings, ref_class_names, idx_to_class


def classify_crop(model, crop_image: np.ndarray, ref_embeddings: np.ndarray,
                  ref_class_names: List[str]) -> Tuple[str, float]:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_embedding = model.get_embeddings(input_tensor).cpu().numpy()

    similarities = np.dot(query_embedding, ref_embeddings.T).flatten()
    top_idx = np.argmax(similarities)
    predicted_class = ref_class_names[top_idx]
    confidence = similarities[top_idx]

    display_name = Path(predicted_class).stem
    return display_name, confidence


def run_detection(model, image_path: str) -> Tuple[List[Dict], float]:
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


def draw_annotated_image(img: np.ndarray, df: pd.DataFrame,
                         class_colors: Dict, mode: str = 'classification') -> np.ndarray:
    """
    Draw bounding boxes on image.
    mode='detection'      → simple colored boxes, no labels
    mode='classification' → PIL-based labels (Turkish char support) inside box
    """
    if mode == 'detection':
        out = img.copy()
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            color = class_colors[row['predicted_class']]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        return out

    # classification mode: PIL with Turkish font
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, 'RGBA')

    try:
        font_name = ImageFont.truetype(FONT_PATH, size=13)
        font_conf = ImageFont.truetype(FONT_PATH, size=11)
    except Exception:
        font_name = ImageFont.load_default()
        font_conf = font_name

    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        pred_class = row['predicted_class']
        conf = row['class_confidence']
        r, g, b = class_colors[pred_class]

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b), width=3)

        name_line = pred_class
        conf_line = f"{conf * 100:.0f}%"

        nb = draw.textbbox((0, 0), name_line, font=font_name)
        cb = draw.textbbox((0, 0), conf_line, font=font_conf)
        text_w = max(nb[2] - nb[0], cb[2] - cb[0]) + 8
        text_h = (nb[3] - nb[1]) + (cb[3] - cb[1]) + 10

        bg_x2 = min(x1 + text_w, x2)
        bg_y2 = min(y1 + text_h, y2)
        draw.rectangle([x1, y1, bg_x2, bg_y2], fill=(0, 0, 0, 140))
        draw.text((x1 + 4, y1 + 3), name_line, font=font_name, fill=(255, 255, 255, 255))
        draw.text((x1 + 4, y1 + 3 + (nb[3] - nb[1]) + 2), conf_line, font=font_conf, fill=(r, g, b, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_image(
    image_path: str,
    detection_model,
    classification_model,
    ref_embeddings: np.ndarray,
    ref_class_names: List[str],
    cls_folder: Path,
    det_folder: Path,
    plan_folder: Path,
) -> Tuple[pd.DataFrame, Dict]:
    timing = {'detection': 0, 'classification': 0, 'total': 0}
    total_start = time.time()

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]

    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARNING] Could not load image: {image_path}")
        return pd.DataFrame(), timing

    img_h, img_w = img.shape[:2]

    detections, det_time = run_detection(detection_model, image_path)
    timing['detection'] = det_time

    if not detections:
        print(f"  No detections in {filename}")
        return pd.DataFrame(), timing

    results = []
    class_colors = {}

    for det in detections:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        cls_start = time.time()
        predicted_class, class_conf = classify_crop(
            classification_model, crop, ref_embeddings, ref_class_names
        )
        timing['classification'] += time.time() - cls_start

        results.append({
            'image_name': filename,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'detection_conf': det['detection_conf'],
            'predicted_class': predicted_class,
            'class_confidence': class_conf,
        })

        if predicted_class not in class_colors:
            class_colors[predicted_class] = get_color_for_class(predicted_class)

    if not results:
        return pd.DataFrame(), timing

    df = pd.DataFrame(results)

    # --- Save CSV → classification/ ---
    df.to_csv(cls_folder / f"{base_name}_results.csv", index=False)

    # --- Detection annotated image → detection/ ---
    det_img = draw_annotated_image(img, df, class_colors, mode='detection')
    cv2.imwrite(str(det_folder / f"{base_name}_detection.jpg"), det_img)

    # --- Classification annotated image → classification/ ---
    cls_img = draw_annotated_image(img, df, class_colors, mode='classification')
    cv2.imwrite(str(cls_folder / f"{base_name}_annotated.jpg"), cls_img)

    # --- Planogram → planogram/ ---
    print(f"  Generating planogram...")
    plan_path = str(plan_folder / f"{base_name}_planogram.png")
    fig = generate_planogram(
        df, img_h, img_w,
        image_path=image_path,
        output_path=plan_path,
        show_images=True,
        title=f"Planogram — {base_name}",
    )
    plt.close(fig)

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
    args = parser.parse_args()

    global CONF_THRESHOLD
    CONF_THRESHOLD = args.conf

    input_folder = Path(args.input)
    output_folder = Path(args.output) if args.output else OUTPUT_FOLDER

    if not input_folder.exists():
        print(f"[ERROR] Input folder not found: {input_folder}")
        return

    cls_folder  = output_folder / "classification"
    det_folder  = output_folder / "detection"
    plan_folder = output_folder / "planogram"
    for folder in [cls_folder, det_folder, plan_folder]:
        folder.mkdir(parents=True, exist_ok=True)

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
            cls_folder,
            det_folder,
            plan_folder,
        )
        all_timings.append(timing)
        if not df.empty:
            all_results.append(df)

    total_pipeline_time = time.time() - total_start

    # 5. Save Combined Results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = cls_folder / "all_results.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to: {combined_path}")

        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)
        class_counts = combined_df['predicted_class'].value_counts()
        for cls, count in class_counts.head(10).items():
            print(f"  {cls}: {count}")
        if len(class_counts) > 10:
            print(f"  ... and {len(class_counts) - 10} more classes")

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

    pathlib.PosixPath = temp
    print(f"\nDone! Check folder: '{output_folder}'")


if __name__ == '__main__':
    main()
