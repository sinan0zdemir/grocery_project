"""
ArcFace Evaluation (v4 Architecture)

Evaluates the ResNet-34 + ArcFace + HAL model trained by arcface_migros_v4.py.
Builds reference DB from Training images, evaluates on Testing shelf crops.

Metrics:
  - HR@1, HR@5, HR@10, HR@20 (name-level retrieval accuracy)
  - mAP (mean Average Precision)
  - Confusion analysis (top confused pairs, worst performing classes)

Usage:
    python evaluate_arcface.py
    python evaluate_arcface.py --checkpoint best.pth
    python evaluate_arcface.py --checkpoint best.pth --top_k 20

Expects:
    - datasets/migros_dataset_v6/Training/           (product images)
    - datasets/migros_dataset_v6/Testing/            (shelf photos)
    - datasets/migros_dataset_v6/Annotations/        (_annotations.csv + SDP mapping)
    - classification/checkpoints/<checkpoint>.pth     (v4 checkpoint)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Turkish -> ASCII for Windows console
_TR_MAP = str.maketrans('\u00e7\u00c7\u011f\u011e\u0131\u0130\u00f6\u00d6\u015f\u015e\u00fc\u00dc', 'cCgGiIoOsSuU')
def safe(s): return str(s).translate(_TR_MAP)

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "migros_dataset_v6"
TRAINING_DIR = DATASET_DIR / "Training"
TESTING_DIR = DATASET_DIR / "Testing"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"
ANNOTATIONS_CSV = ANNOTATIONS_DIR / "_annotations.csv"
PRODUCT_MAP_CSV = ANNOTATIONS_DIR / "SDP_Product&ID_Dataset_fix.csv"

CHECKPOINT_DIR = SCRIPT_DIR.parent / "checkpoints"
DEFAULT_CHECKPOINT = "best_newest.pth"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "arcface_v4"

FEATURE_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading
# =============================================================================
def load_product_id_mapping(annotations_dir: Path) -> Dict[int, str]:
    """Load product ID to name mapping from SDP_Product&ID_Dataset_fix.csv"""
    mapping_file = annotations_dir / "SDP_Product&ID_Dataset_fix.csv"
    id_to_name = {}
    with open(mapping_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                try:
                    product_id = int(parts[0])
                    product_name = parts[1].strip()
                    id_to_name[product_id] = product_name
                except ValueError:
                    continue
    return id_to_name


def normalize_product_name(name: str) -> str:
    """Normalize product name for matching."""
    name = os.path.splitext(name)[0]
    return name.lower().strip()


def parse_annotations(annotations_dir: Path, testing_dir: Path, id_to_name: Dict) -> List[Dict]:
    """Parse _annotations.csv for test samples."""
    ann_file = annotations_dir / "_annotations.csv"
    df = pd.read_csv(ann_file)
    samples = []

    for _, row in df.iterrows():
        filename = row["filename"]
        product_id = int(row["class"])
        bbox = [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])]

        image_path = testing_dir / filename
        if image_path.exists() and product_id in id_to_name:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                samples.append({
                    "image_path": str(image_path),
                    "product_id": product_id,
                    "product_name": id_to_name[product_id],
                    "bbox": bbox,
                })

    print(f"  Parsed {len(samples)} test samples from {df['filename'].nunique()} images")
    return samples


def build_training_set(training_dir: Path, id_to_name: Dict, valid_product_ids=None):
    """
    Build training set from hierarchical folder structure.
    Uses product NAME as class identifier (not ID).
    Returns class_to_idx, idx_to_class, image_paths.
    """
    name_to_id = {}
    for pid, pname in id_to_name.items():
        norm = normalize_product_name(pname)
        name_to_id[norm] = pid

    image_paths = []
    class_to_idx = {}
    idx_to_class = {}

    all_images = []
    for root, dirs, files in os.walk(str(training_dir)):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                all_images.append((full_path, f))

    for full_path, filename in all_images:
        norm_filename = normalize_product_name(filename)

        product_id = None
        for norm_name, pid in name_to_id.items():
            if norm_name == norm_filename or norm_name in norm_filename or norm_filename in norm_name:
                product_id = pid
                break

        if product_id is None:
            continue
        if valid_product_ids is not None and product_id not in valid_product_ids:
            continue

        class_name = id_to_name[product_id]
        if class_name not in class_to_idx:
            idx = len(class_to_idx)
            class_to_idx[class_name] = idx
            idx_to_class[idx] = class_name

        image_paths.append((full_path, class_to_idx[class_name], class_name))

    print(f"  {len(image_paths)} training images, {len(class_to_idx)} classes")
    return class_to_idx, idx_to_class, image_paths


def build_test_samples(raw_samples, class_to_idx, id_to_name):
    """Match test samples to training class indices using product name."""
    matched = []
    unmatched_names = set()

    for s in raw_samples:
        class_name = id_to_name.get(s["product_id"], "")
        if class_name in class_to_idx:
            matched.append({
                "image_path": s["image_path"],
                "bbox": s["bbox"],
                "class_idx": class_to_idx[class_name],
                "class_name": class_name,
            })
        else:
            unmatched_names.add(class_name)

    if unmatched_names:
        print(f"  WARNING: {len(unmatched_names)} products not in training set:")
        for n in sorted(unmatched_names):
            print(f"    - {safe(n)}")

    print(f"  {len(matched)} matched test samples")
    return matched


# =============================================================================
# Datasets
# =============================================================================
class TestDataset(Dataset):
    """Crops bounding boxes from shelf images."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s["image_path"]

        try:
            if img_path not in self._cache:
                self._cache[img_path] = Image.open(img_path).convert("RGB")
            image = self._cache[img_path]

            x1, y1, x2, y2 = s["bbox"]
            w, h = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        return self.transform(image), s["class_idx"]


# =============================================================================
# Model (matching arcface_migros_v4 architecture)
# =============================================================================
class ProductRecognitionModel(nn.Module):
    """
    ResNet-34 backbone + embedding layer + ArcFace head.
    Must match the architecture from arcface_migros_v4.py for weight loading.
    """
    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()

        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # ArcFace head (needed for weight loading, not used at eval)
        self.arcface = ArcFaceHead(embedding_dim, num_classes, scale, margin)
        self.embedding_dim = embedding_dim

    def get_embeddings(self, x):
        features = self.backbone(x).flatten(1)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, dim=1)

    def forward(self, x):
        return self.get_embeddings(x)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.current_margin = margin
        self.num_classes = num_classes

    def set_margin(self, m):
        self.current_margin = m

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.scale

    def get_proxies(self):
        return F.normalize(self.weight, dim=1)


# =============================================================================
# Reference DB & Evaluation
# =============================================================================
class AutoCropWhiteBackground:
    """Auto-detect product bounds by finding non-white region."""
    def __init__(self, threshold=240, padding=20, max_size=(1400, 700)):
        self.threshold = threshold
        self.padding = padding
        self.max_size = max_size

    def __call__(self, img):
        img_array = np.array(img)
        non_white = ~np.all(img_array > self.threshold, axis=2)

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)

        if not np.any(rows) or not np.any(cols):
            return transforms.CenterCrop(self.max_size)(img)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        h, w = img_array.shape[:2]
        y_min = max(0, y_min - self.padding)
        y_max = min(h, y_max + self.padding)
        x_min = max(0, x_min - self.padding)
        x_max = min(w, x_max + self.padding)

        cropped = img.crop((x_min, y_min, x_max, y_max))
        crop_w, crop_h = cropped.size
        max_h, max_w = self.max_size
        if crop_h > max_h or crop_w > max_w:
            cropped = transforms.CenterCrop(self.max_size)(cropped)
        return cropped


@torch.no_grad()
def build_reference_db(model, image_paths, transform, device):
    """Build reference embeddings from training images."""
    model.eval()
    embeddings, labels = [], []

    print("  Building reference DB...")
    for path, class_idx, _ in tqdm(image_paths, desc="Ref DB"):
        try:
            image = Image.open(path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            emb = model.get_embeddings(image).cpu().numpy()
            embeddings.append(emb)
            labels.append(class_idx)
        except Exception:
            pass

    return np.vstack(embeddings), np.array(labels)


@torch.no_grad()
def evaluate(model, test_loader, ref_emb, ref_lbl, device, idx_to_class, top_k=20):
    """Evaluate with HR@K, mAP, and confusion analysis."""
    model.eval()

    query_emb, query_lbl = [], []
    for images, labels in tqdm(test_loader, desc="Eval"):
        emb = model.get_embeddings(images.to(device)).cpu().numpy()
        query_emb.append(emb)
        query_lbl.append(labels.numpy())

    query_emb = np.vstack(query_emb)
    query_lbl = np.concatenate(query_lbl)

    sims = query_emb @ ref_emb.T
    sorted_idx = np.argsort(-sims, axis=1)

    # HR@K
    results = {}
    for k in [1, 5, 10, 20]:
        if k > top_k:
            break
        hits = sum(q in ref_lbl[sorted_idx[i, :k]] for i, q in enumerate(query_lbl))
        results[f"HR@{k}"] = 100 * hits / len(query_lbl)

    # mAP
    aps = []
    for i, q in enumerate(query_lbl):
        relevant = ref_lbl[sorted_idx[i]] == q
        if relevant.sum() > 0:
            prec = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
            aps.append((prec * relevant).sum() / relevant.sum())
    results["mAP"] = 100 * np.mean(aps) if aps else 0

    # Confusion analysis
    confusion_pairs = defaultdict(int)
    class_totals = defaultdict(int)
    class_correct = defaultdict(int)

    for i, true_lbl in enumerate(query_lbl):
        pred_lbl = ref_lbl[sorted_idx[i, 0]]
        class_totals[true_lbl] += 1
        if pred_lbl == true_lbl:
            class_correct[true_lbl] += 1
        else:
            confusion_pairs[(true_lbl, pred_lbl)] += 1

    sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: -x[1])
    results["confusion"] = {
        "pairs": sorted_confusion,
        "class_totals": dict(class_totals),
        "class_correct": dict(class_correct),
    }

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ArcFace v4 Evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint filename in checkpoints/ dir")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_DIR / args.checkpoint

    print("=" * 70)
    print("ArcFace v4 Evaluation - Migros v6 Dataset")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Validate paths
    for p, n in [(TRAINING_DIR, "Training"), (TESTING_DIR, "Testing"),
                  (ANNOTATIONS_CSV, "Annotations"), (checkpoint_path, "Checkpoint")]:
        if not p.exists():
            print(f"  ERROR: {n} not found: {p}")
            return
        print(f"  [OK] {n}")

    # =========================================================================
    # 1. Load checkpoint FIRST to get class mapping and num_classes
    # =========================================================================
    print(f"\n1. Loading checkpoint ({args.checkpoint})...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if "class_to_idx" in ckpt:
        class_to_idx = ckpt["class_to_idx"]
        idx_to_class = ckpt.get("idx_to_class", {v: k for k, v in class_to_idx.items()})
        NUM_CLASSES = ckpt.get("num_classes", len(class_to_idx))
        print(f"  Checkpoint classes: {NUM_CLASSES}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
    else:
        print("  ERROR: Checkpoint has no class_to_idx! Cannot determine class mapping.")
        return

    # Load id_to_name from checkpoint if available, else from CSV
    if "id_to_name" in ckpt:
        id_to_name = ckpt["id_to_name"]
        print(f"  Product mapping from checkpoint: {len(id_to_name)} entries")
    else:
        id_to_name = load_product_id_mapping(ANNOTATIONS_DIR)
        print(f"  Product mapping from CSV: {len(id_to_name)} entries")

    # =========================================================================
    # 2. Build model with correct num_classes and load weights
    # =========================================================================
    print(f"\n2. Loading model...")
    model = ProductRecognitionModel(
        num_classes=NUM_CLASSES,
        embedding_dim=FEATURE_DIM,
    ).to(DEVICE)

    state_dict = ckpt["model_state_dict"]
    # Filter out keys that don't exist in our model (e.g. HAL head)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    extra = set(state_dict.keys()) - model_keys
    if extra:
        print(f"  Skipped {len(extra)} checkpoint keys (HAL head etc)")

    # Check shape compatibility
    shape_mismatch = []
    for k, v in filtered.items():
        expected = model.state_dict()[k].shape
        if v.shape != expected:
            shape_mismatch.append(f"  {k}: ckpt={v.shape} model={expected}")
    if shape_mismatch:
        print(f"  ERROR: {len(shape_mismatch)} shape mismatches:")
        for s in shape_mismatch:
            print(s)
        return

    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"  Model loaded: {NUM_CLASSES} classes, {FEATURE_DIM}d embeddings")

    # =========================================================================
    # 3. Load test annotations and build training set using checkpoint mapping
    # =========================================================================
    print(f"\n3. Loading data...")
    raw_samples = parse_annotations(ANNOTATIONS_DIR, TESTING_DIR, id_to_name)

    # Build test samples using the CHECKPOINT's class_to_idx
    test_samples = build_test_samples(raw_samples, class_to_idx, id_to_name)

    # Build training set — remap to checkpoint class indices
    annotation_product_ids = set(s["product_id"] for s in raw_samples)
    _, _, image_paths_raw = build_training_set(
        TRAINING_DIR, id_to_name, valid_product_ids=annotation_product_ids
    )

    # Re-map image_paths to use checkpoint's class_to_idx
    image_paths = []
    for path, _, class_name in image_paths_raw:
        if class_name in class_to_idx:
            image_paths.append((path, class_to_idx[class_name], class_name))
    print(f"  Remapped {len(image_paths)} training images to checkpoint class indices")

    # =========================================================================
    # 4. Transforms
    # =========================================================================
    # Reference DB: auto-crop white backgrounds (training product images)
    ref_transform = transforms.Compose([
        AutoCropWhiteBackground(threshold=240, padding=20, max_size=(1400, 700)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Test: direct resize (shelf crops, no white bg)
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # =========================================================================
    # 5. Build reference DB from training images
    # =========================================================================
    print("\n4. Building reference database...")
    t0 = time.time()
    ref_emb, ref_lbl = build_reference_db(model, image_paths, ref_transform, DEVICE)
    print(f"  {len(ref_emb)} embeddings, {len(set(ref_lbl))} classes ({time.time()-t0:.1f}s)")

    # =========================================================================
    # 6. Evaluate
    # =========================================================================
    test_dataset = TestDataset(test_samples, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n5. Evaluating ({len(test_samples)} crops)...")
    t0 = time.time()
    results = evaluate(model, test_loader, ref_emb, ref_lbl, DEVICE, idx_to_class, args.top_k)
    eval_time = time.time() - t0

    # =========================================================================
    # 7. Print results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test crops: {len(test_samples)}")
    print(f"  Reference DB: {len(ref_emb)} embeddings")
    print(f"  Eval time: {eval_time:.1f}s")
    print()

    print("  +------------------+-----------+")
    print("  | Metric           | Value     |")
    print("  +------------------+-----------+")
    for k in [1, 5, 10, 20]:
        key = f"HR@{k}"
        if key in results:
            print(f"  | {key:<16} | {results[key]:8.2f}% |")
    print(f"  | {'mAP':<16} | {results['mAP']:8.2f}% |")
    print("  +------------------+-----------+")

    # Confusion analysis
    confusion = results.get("confusion", {})
    pairs = confusion.get("pairs", [])
    class_totals = confusion.get("class_totals", {})
    class_correct = confusion.get("class_correct", {})

    if pairs:
        total_errors = sum(c for _, c in pairs)
        print(f"\n  TOP CONFUSED PAIRS ({total_errors} total errors):")
        print("  " + "-" * 65)

        for i, ((true_cls, pred_cls), count) in enumerate(pairs[:15]):
            true_name = safe(idx_to_class.get(true_cls, f"?{true_cls}"))
            pred_name = safe(idx_to_class.get(pred_cls, f"?{pred_cls}"))
            pct = 100 * count / class_totals.get(true_cls, 1)
            print(f"  {i+1:2d}. {true_name[:35]:<35} -> {pred_name[:30]}")
            print(f"      {count} errors ({pct:.1f}% of class)")

    # Per-class accuracy (worst first)
    class_acc = {}
    for cls_idx in class_totals:
        total = class_totals[cls_idx]
        correct = class_correct.get(cls_idx, 0)
        class_acc[cls_idx] = (100 * correct / total, correct, total)

    print(f"\n  WORST PERFORMING CLASSES:")
    print("  " + "-" * 55)
    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1][0])
    for cls_idx, (acc, correct, total) in sorted_acc[:15]:
        cls_name = safe(idx_to_class.get(cls_idx, f"?{cls_idx}"))
        print(f"  {acc:5.1f}% ({correct:2d}/{total:2d}) - {cls_name[:50]}")

    # All classes
    print(f"\n  ALL CLASSES (by accuracy):")
    print("  " + "-" * 55)
    for cls_idx, (acc, correct, total) in sorted(class_acc.items(), key=lambda x: -x[1][0]):
        cls_name = safe(idx_to_class.get(cls_idx, f"?{cls_idx}"))
        print(f"  {acc:5.1f}% ({correct:2d}/{total:2d}) - {cls_name[:50]}")

    # =========================================================================
    # 8. Save outputs
    # =========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "eval_results.txt", "w", encoding="utf-8") as f:
        f.write(f"ArcFace v4 Evaluation - Migros v6\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test crops: {len(test_samples)}\n")
        f.write(f"Training classes: {NUM_CLASSES}\n")
        f.write(f"Reference DB: {len(ref_emb)} embeddings\n\n")

        f.write("--- Retrieval Metrics ---\n")
        for k in [1, 5, 10, 20]:
            key = f"HR@{k}"
            if key in results:
                f.write(f"{key}: {results[key]:.2f}%\n")
        f.write(f"mAP: {results['mAP']:.2f}%\n\n")

        f.write("--- Per-Class Accuracy ---\n")
        for cls_idx, (acc, correct, total) in sorted(class_acc.items(), key=lambda x: -x[1][0]):
            cls_name = idx_to_class.get(cls_idx, f"?{cls_idx}")
            f.write(f"  {acc:5.1f}% ({correct:2d}/{total:2d}) - {cls_name}\n")

        if pairs:
            f.write(f"\n--- Top Confused Pairs ---\n")
            for (true_cls, pred_cls), count in pairs[:20]:
                true_name = idx_to_class.get(true_cls, f"?{true_cls}")
                pred_name = idx_to_class.get(pred_cls, f"?{pred_cls}")
                f.write(f"  {true_name} -> {pred_name}: {count}\n")

    print(f"\n  Results: {OUTPUT_DIR / 'eval_results.txt'}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
