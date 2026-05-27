"""
ArcFace Evaluation on Migros v6 Dataset

Evaluates the augmented ResNet50-ArcFace model on the Migros v6 test set.
Uses the pre-built reference database (reference_db_new.pt).

Test images are full shelf photos; annotations provide bounding boxes with
numeric class IDs that map to dataset_arcface folder names.

Usage:
    python evaluate_arcface_migros.py

Expects:
    - datasets/migros_dataset_v6/Testing/           (5 shelf images)
    - datasets/migros_dataset_v6/Annotations/       (_annotations.csv + SDP mapping)
    - classification/checkpoints/augmented_resnet50_arcface.pth
    - classification/eval/outputs/reference_db_new.pt  (pre-built)
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "migros_dataset_v6"
TESTING_DIR = DATASET_DIR / "Testing"
ANNOTATIONS_CSV = DATASET_DIR / "Annotations" / "_annotations.csv"
PRODUCT_MAP_CSV = DATASET_DIR / "Annotations" / "SDP_Product&ID_Dataset_fix.csv"

CHECKPOINT_PATH = SCRIPT_DIR.parent / "checkpoints" / "augmented_resnet50_arcface.pth"
REFERENCE_DB_PATH = SCRIPT_DIR / "outputs" / "reference_db_new.pt"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "migros_v6"

EMBEDDING_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 10
NUM_VIZ = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading
# =============================================================================
def load_product_mapping(csv_path: Path) -> Dict[str, str]:
    """Load numeric class ID -> product name mapping."""
    mapping = {}
    if not csv_path.exists():
        print(f"WARNING: Product mapping not found at {csv_path}")
        return mapping
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def load_annotations(csv_path: Path, testing_dir: Path) -> List[Dict]:
    """
    Parse _annotations.csv and resolve image paths.
    Returns list of {image_path, class_id, bbox}.
    """
    df = pd.read_csv(csv_path)
    samples = []
    missing_images = set()

    for _, row in df.iterrows():
        filename = row["filename"]
        class_id = str(int(row["class"]))
        bbox = [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])]

        # Resolve image path
        img_path = testing_dir / filename
        if not img_path.exists():
            missing_images.add(filename)
            continue

        samples.append({
            "image_path": str(img_path),
            "class_id": class_id,
            "bbox": bbox,
            "filename": filename
        })

    if missing_images:
        print(f"WARNING: {len(missing_images)} images not found in {testing_dir}")
        for m in list(missing_images)[:3]:
            print(f"  - {m}")

    print(f"Loaded {len(samples)} annotated crops from {df['filename'].nunique()} images")
    return samples


class CropDataset(Dataset):
    """Dataset that crops bounding boxes from shelf images."""
    
    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform
        # Cache loaded images to avoid re-reading the same shelf image
        self._image_cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        class_id = sample["class_id"]
        x1, y1, x2, y2 = sample["bbox"]

        # Load and cache full image
        if img_path not in self._image_cache:
            self._image_cache[img_path] = Image.open(img_path).convert("RGB")
        image = self._image_cache[img_path]

        # Crop
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crop = image.crop((x1, y1, x2, y2))
        else:
            crop = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            crop = self.transform(crop)

        return crop, class_id, idx


# =============================================================================
# Model
# =============================================================================
def load_model(checkpoint_path: Path) -> nn.Module:
    """Load the augmented ResNet50 ArcFace backbone."""
    print(f"Loading model from {checkpoint_path}")
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, EMBEDDING_DIM)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(state_dict, strict=True)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    print(f"Model loaded on {DEVICE}")
    return backbone


def load_reference_db(db_path: Path):
    """Load pre-built reference embeddings."""
    print(f"Loading reference DB from {db_path}")
    data = torch.load(db_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]  # numpy array [N, 512]
    class_names = data["class_names"]  # list of str (folder names = class IDs)
    labels = data["labels"]
    print(f"Reference DB: {len(embeddings)} embeddings, {len(set(class_names))} classes")
    return embeddings, class_names, labels


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    ref_embeddings: np.ndarray,
    ref_class_names: List[str],
    samples: List[Dict],
    product_map: Dict[str, str],
    top_k: int = TOP_K,
) -> Tuple[Dict, List[Dict]]:
    """Run retrieval evaluation."""

    all_results = []
    correct_id = {1: 0, 5: 0, 10: 0}   # exact class-ID match
    correct_name = {1: 0, 5: 0, 10: 0}  # product-name match (tolerates variant IDs)
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    name_correct = defaultdict(int)
    name_total = defaultdict(int)

    print("Evaluating...")
    with torch.no_grad():
        for batch_imgs, batch_class_ids, batch_indices in tqdm(dataloader, desc="Eval"):
            batch_imgs = batch_imgs.to(DEVICE)

            # Forward pass
            embeddings = model(batch_imgs)
            embeddings = F.normalize(embeddings, dim=1).cpu().numpy()

            # Cosine similarity against reference DB
            similarities = np.dot(embeddings, ref_embeddings.T)

            for i in range(len(batch_imgs)):
                query_class = batch_class_ids[i]
                query_name = product_map.get(query_class, query_class)
                sample_idx = batch_indices[i].item()

                # Top-k retrieval
                top_k_indices = np.argsort(similarities[i])[::-1][:top_k]
                top_k_classes = [ref_class_names[j] for j in top_k_indices]
                top_k_scores = [float(similarities[i][j]) for j in top_k_indices]
                top_k_names = [product_map.get(c, c) for c in top_k_classes]

                total += 1
                class_total[query_class] += 1
                name_total[query_name] += 1

                # Exact ID match
                is_top1_id = top_k_classes[0] == query_class
                if is_top1_id:
                    correct_id[1] += 1
                    class_correct[query_class] += 1
                if query_class in top_k_classes[:5]:
                    correct_id[5] += 1
                if query_class in top_k_classes[:10]:
                    correct_id[10] += 1

                # Name-level match (correct product, possibly different variant/size)
                is_top1_name = top_k_names[0] == query_name
                if is_top1_name:
                    correct_name[1] += 1
                    name_correct[query_name] += 1
                if query_name in top_k_names[:5]:
                    correct_name[5] += 1
                if query_name in top_k_names[:10]:
                    correct_name[10] += 1

                all_results.append({
                    "sample_idx": sample_idx,
                    "query_class": query_class,
                    "query_name": query_name,
                    "top_k_classes": top_k_classes,
                    "top_k_names": top_k_names,
                    "top_k_scores": top_k_scores,
                    "is_correct_id": is_top1_id,
                    "is_correct_name": is_top1_name,
                    "bbox": samples[sample_idx]["bbox"],
                    "image_path": samples[sample_idx]["image_path"],
                })

    metrics = {}
    for k in [1, 5, 10]:
        if k <= top_k:
            metrics[f"top{k}_id_accuracy"] = correct_id[k] / total if total > 0 else 0
            metrics[f"top{k}_name_accuracy"] = correct_name[k] / total if total > 0 else 0
    metrics["total_queries"] = total
    metrics["num_classes_tested"] = len(class_total)
    metrics["num_products_tested"] = len(name_total)

    # Per-class accuracy (by exact ID)
    per_class = {}
    for cls in sorted(class_total.keys(), key=lambda x: int(x)):
        acc = class_correct[cls] / class_total[cls]
        name = product_map.get(cls, cls)
        per_class[cls] = {
            "name": name,
            "correct": class_correct[cls],
            "total": class_total[cls],
            "accuracy": acc,
        }
    metrics["per_class"] = per_class

    # Per-product accuracy (by name, merging duplicate variant IDs)
    per_product = {}
    for name in sorted(name_total.keys()):
        acc = name_correct[name] / name_total[name]
        per_product[name] = {
            "correct": name_correct[name],
            "total": name_total[name],
            "accuracy": acc,
        }
    metrics["per_product"] = per_product

    return metrics, all_results


# =============================================================================
# Confusion Analysis
# =============================================================================
def print_confusion_analysis(results: List[Dict], product_map: Dict[str, str]):
    """Print most common misclassifications."""
    misclass = defaultdict(lambda: Counter())
    for r in results:
        if not r["is_correct_name"]:
            pred_name = product_map.get(r["top_k_classes"][0], r["top_k_classes"][0])
            gt_name = r["query_name"]
            misclass[gt_name][pred_name] += 1

    if not misclass:
        print("\nNo misclassifications!")
        return

    print("\n" + "=" * 60)
    print("TOP MISCLASSIFICATIONS")
    print("=" * 60)

    # Sort by total errors
    sorted_classes = sorted(misclass.items(), key=lambda x: sum(x[1].values()), reverse=True)
    for gt_name, preds in sorted_classes[:15]:
        total_err = sum(preds.values())
        top_preds = preds.most_common(3)
        pred_str = ", ".join(f"{p} ({c}x)" for p, c in top_preds)
        print(f"  {gt_name} [{total_err} errors] → {pred_str}")


# =============================================================================
# Visualization
# =============================================================================
def visualize_results(results: List[Dict], samples: List[Dict], 
                      ref_db_path: Path, output_dir: Path, num_examples: int = NUM_VIZ):
    """Save visualization grids: query crop + top-5 matches."""
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = [r for r in results if r["is_correct_name"]]
    incorrect = [r for r in results if not r["is_correct_name"]]

    selected = []
    n_correct = min(len(correct), num_examples // 2)
    n_incorrect = min(len(incorrect), num_examples - n_correct)
    if correct:
        selected.extend(random.sample(correct, n_correct))
    if incorrect:
        selected.extend(random.sample(incorrect, n_incorrect))

    print(f"Generating {len(selected)} visualizations...")

    for viz_idx, result in enumerate(selected):
        fig, axes = plt.subplots(1, 6, figsize=(20, 4))
        
        # Query crop
        try:
            img = Image.open(result["image_path"]).convert("RGB")
            x1, y1, x2, y2 = result["bbox"]
            w, h = img.size
            crop = img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
            axes[0].imshow(crop)
            status = "✓ CORRECT" if result["is_correct_name"] else "✗ WRONG"
            color = "green" if result["is_correct_name"] else "red"
            axes[0].set_title(f"Query: {result['query_name']}\n{status}", 
                            fontsize=8, color=color, fontweight="bold")
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error", ha="center")
        axes[0].axis("off")

        # Top-5 predictions
        for i in range(min(5, len(result["top_k_classes"]))):
            ax = axes[i + 1]
            pred_name = result["top_k_names"][i]
            score = result["top_k_scores"][i]
            match = result["top_k_classes"][i] == result["query_class"]
            color = "green" if match else "red"
            ax.text(0.5, 0.5, f"{pred_name}\n\nScore: {score:.3f}",
                   ha="center", va="center", fontsize=7, transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor="lightgreen" if match else "lightyellow"))
            ax.set_title(f"Top-{i+1}", fontsize=9, color=color)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"migros_viz_{viz_idx}.png", bbox_inches="tight", dpi=150)
        plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("ArcFace Evaluation — Migros v6 Dataset")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 1. Validate paths
    for path, name in [
        (TESTING_DIR, "Testing"),
        (ANNOTATIONS_CSV, "Annotations CSV"),
        (CHECKPOINT_PATH, "Checkpoint"),
        (REFERENCE_DB_PATH, "Reference DB"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            return
        print(f"  ✓ {name}: {path}")

    # 2. Load data
    print("\nStep 1: Loading annotations...")
    product_map = load_product_mapping(PRODUCT_MAP_CSV)
    print(f"  Product mapping: {len(product_map)} entries")

    samples = load_annotations(ANNOTATIONS_CSV, TESTING_DIR)
    if not samples:
        print("ERROR: No valid samples found")
        return

    # 3. Load model & reference DB
    print("\nStep 2: Loading model...")
    model = load_model(CHECKPOINT_PATH)

    print("\nStep 3: Loading reference DB (pre-built)...")
    ref_embeddings, ref_class_names, ref_labels = load_reference_db(REFERENCE_DB_PATH)

    # Verify class overlap
    ref_classes = set(ref_class_names)
    test_classes = set(s["class_id"] for s in samples)
    overlap = ref_classes & test_classes
    missing_from_ref = test_classes - ref_classes
    print(f"  Reference classes: {len(ref_classes)}")
    print(f"  Test classes: {len(test_classes)}")
    print(f"  Overlap: {len(overlap)}")
    if missing_from_ref:
        print(f"  WARNING: {len(missing_from_ref)} test classes missing from ref DB: {sorted(missing_from_ref, key=int)[:10]}")

    # 4. Build dataloader
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = CropDataset(samples, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5. Evaluate
    print("\nStep 4: Running evaluation...")
    metrics, results = evaluate(model, dataloader, ref_embeddings, ref_class_names, 
                                 samples, product_map)

    # 6. Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total test crops    : {metrics['total_queries']}")
    print(f"  Class IDs tested    : {metrics['num_classes_tested']}")
    print(f"  Unique products     : {metrics['num_products_tested']}")
    print()
    print(f"  --- Exact ID Match ---")
    print(f"  Top-1 Accuracy      : {metrics['top1_id_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy      : {metrics['top5_id_accuracy'] * 100:.2f}%")
    print(f"  Top-10 Accuracy     : {metrics['top10_id_accuracy'] * 100:.2f}%")
    print()
    print(f"  --- Product Name Match (tolerates variant IDs) ---")
    print(f"  Top-1 Accuracy      : {metrics['top1_name_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Accuracy      : {metrics['top5_name_accuracy'] * 100:.2f}%")
    print(f"  Top-10 Accuracy     : {metrics['top10_name_accuracy'] * 100:.2f}%")

    # Per-product summary: worst performers
    print("\n" + "-" * 60)
    print("WORST PERFORMING PRODUCTS (by name-level accuracy)")
    print("-" * 60)
    per_product = metrics["per_product"]
    sorted_products = sorted(per_product.items(), key=lambda x: x[1]["accuracy"])
    for name, entry in sorted_products[:15]:
        print(f"  {entry['accuracy']*100:5.1f}%  ({entry['correct']}/{entry['total']})  {name}")

    # Confusion analysis
    print_confusion_analysis(results, product_map)

    # 7. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_file = OUTPUT_DIR / "migros_v6_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("ArcFace Evaluation — Migros v6 Dataset\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total test crops: {metrics['total_queries']}\n")
        f.write(f"Class IDs tested: {metrics['num_classes_tested']}\n")
        f.write(f"Unique products: {metrics['num_products_tested']}\n\n")
        f.write(f"--- Exact ID Match ---\n")
        f.write(f"Top-1: {metrics['top1_id_accuracy']:.4f} ({metrics['top1_id_accuracy']*100:.2f}%)\n")
        f.write(f"Top-5: {metrics['top5_id_accuracy']:.4f} ({metrics['top5_id_accuracy']*100:.2f}%)\n")
        f.write(f"Top-10: {metrics['top10_id_accuracy']:.4f} ({metrics['top10_id_accuracy']*100:.2f}%)\n\n")
        f.write(f"--- Product Name Match ---\n")
        f.write(f"Top-1: {metrics['top1_name_accuracy']:.4f} ({metrics['top1_name_accuracy']*100:.2f}%)\n")
        f.write(f"Top-5: {metrics['top5_name_accuracy']:.4f} ({metrics['top5_name_accuracy']*100:.2f}%)\n")
        f.write(f"Top-10: {metrics['top10_name_accuracy']:.4f} ({metrics['top10_name_accuracy']*100:.2f}%)\n")
        f.write("\nPer-Product Breakdown (name-level):\n")
        f.write("-" * 50 + "\n")
        for name, entry in sorted(per_product.items(), key=lambda x: x[1]["accuracy"]):
            f.write(f"  {entry['accuracy']*100:5.1f}%  ({entry['correct']}/{entry['total']})  {name}\n")

    print(f"\nResults saved to {results_file}")

    # 8. Visualizations
    visualize_results(results, samples, REFERENCE_DB_PATH, OUTPUT_DIR)
    print(f"Visualizations saved to {OUTPUT_DIR}")
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
