"""
ArcFace Grocery Products Evaluation Script (Fixed Version)

This script evaluates a trained ArcFace model on the Grocery_products dataset.
It matches the "fixed" training notebook logic by:
1. Parsing annotations to identify the exact subset of classes used.
2. Building the reference database ONLY from those classes.
3. Matching test images to classes using the specific logic (filename/parent) from the notebook.
4. Using the exact model architecture to ensure state dict compatibility.

Usage:
    python evaluate_arcface.py

Expects:
    - datasets/Grocery_products/Training
    - datasets/Grocery_products/Testing
    - datasets/Grocery_products/Annotations
    - classification/checkpoints/best.pth
"""

import os
import sys
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "Grocery_products"
CHECKPOINT_DIR = SCRIPT_DIR.parent / "checkpoints"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

TRAINING_DIR = DATASET_DIR / "Training"
TESTING_DIR = DATASET_DIR / "Testing"
ANNOTATIONS_DIR = DATASET_DIR / "Annotations"

# Model configuration
EMBEDDING_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 5
NUM_EXAMPLE_IMAGES = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Helper Functions (Ported from Notebook)
# =============================================================================
def normalize_path(path: str) -> str:
    """Normalize path: forward slashes, lowercase extension."""
    path = path.replace('\\', '/')
    # Normalize extension to lowercase
    base, ext = os.path.splitext(path)
    return base + ext.lower()


def parse_all_annotations(annotations_dir: Path, testing_dir: Path) -> Tuple[List[Dict], set]:
    """Parse all annotation files to find valid test samples and classes."""
    all_samples = []
    referenced_classes = set()

    if not annotations_dir.exists():
        print(f"ERROR: {annotations_dir} not found")
        return [], set()

    ann_files = sorted([f.name for f in annotations_dir.iterdir() if f.name.endswith('.csv')])
    print(f"Found {len(ann_files)} annotation files")

    for ann_file in ann_files:
        match = re.match(r's(\d+)_(\d+)\.csv', ann_file)
        if not match:
            continue

        store_num, image_id = match.groups()

        # Find test image (supports both naming formats)
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            # Try with store prefix first (e.g., store1_10.jpg)
            candidate = testing_dir / f'store{store_num}' / 'images' / f'store{store_num}_{image_id}{ext}'
            if candidate.exists():
                image_path = str(candidate)
                break
            # Fallback to without prefix (e.g., 10.jpg)
            candidate = testing_dir / f'store{store_num}' / 'images' / f'{image_id}{ext}'
            if candidate.exists():
                image_path = str(candidate)
                break

        if not image_path:
            continue

        # Parse annotations
        with open(annotations_dir / ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = re.split(r'[\t,]+', line)
                if len(parts) < 5:
                    continue

                class_path = parts[0].strip()
                class_path = normalize_path(class_path)
                if class_path.startswith('/'):
                    class_path = class_path[1:]

                try:
                    coords = []
                    for p in parts[1:5]:
                        p = re.sub(r'[\[\]\s]', '', p)
                        coords.append(int(float(p)))

                    if len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        if x2 > x1 and y2 > y1:
                            all_samples.append({
                                'image_path': image_path,
                                'class_path': class_path,
                                'bbox': [x1, y1, x2, y2]
                            })
                            referenced_classes.add(class_path)
                except (ValueError, IndexError):
                    continue

    print(f"Parsed {len(all_samples)} test samples")
    print(f"Found {len(referenced_classes)} unique classes in annotations")
    
    return all_samples, referenced_classes


def build_training_set(training_dir: Path, valid_classes: Optional[set] = None):
    """
    Build training set with case-insensitive extension matching.
    Returns: class_to_idx, idx_to_class, class_to_category, category_to_idx, image_paths
    """
    image_paths = []
    class_to_idx = {}
    idx_to_class = {}
    class_to_category = {}
    categories = set()

    # Find all training images
    all_images = []
    for root, dirs, files in os.walk(training_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, training_dir)
                rel_path = rel_path.replace('\\', '/')
                all_images.append((full_path, rel_path))

    print(f"Total images in Training folder: {len(all_images)}")

    # Build normalized lookup for valid classes (lowercase with .jpg extension)
    valid_normalized = {}
    if valid_classes:
        for vc in valid_classes:
            # Normalize: lowercase extension
            norm = normalize_path(vc)
            valid_normalized[norm] = vc
        
        unmatched = set(valid_classes)

    matched = 0

    for full_path, rel_path in all_images:
        if valid_classes is not None:
            # Normalize training path
            norm_rel = normalize_path(rel_path)

            # Check if normalized path matches
            if norm_rel in valid_normalized:
                original_ann_path = valid_normalized[norm_rel]
                unmatched.discard(original_ann_path)
                matched += 1
            else:
                continue

        # Use normalized path as class name for consistency
        class_name = normalize_path(rel_path)

        # Get category (top-level folder)
        parts = class_name.split('/')
        category = parts[0] if len(parts) > 1 else 'Unknown'
        categories.add(category)

        if class_name not in class_to_idx:
            idx = len(class_to_idx)
            class_to_idx[class_name] = idx
            idx_to_class[idx] = class_name
            class_to_category[class_name] = category

        image_paths.append((full_path, class_to_idx[class_name], class_name))

    if valid_classes:
        print(f"Matched {matched} classes from annotations")
        if unmatched:
            print(f"WARNING: {len(unmatched)} annotation classes not found!")

    category_to_idx = {cat: idx for idx, cat in enumerate(sorted(categories))}

    return class_to_idx, idx_to_class, class_to_category, category_to_idx, image_paths


def build_test_samples(raw_samples: List[Dict], class_to_idx: Dict[str, int]) -> List[Dict]:
    """
    Match test samples to training class indices.
    """
    matched_samples = []
    unmatched = 0

    # Build reverse lookup for flexible matching
    filename_to_class = {}
    for cls_name in class_to_idx:
        # Store by full path
        filename_to_class[cls_name] = cls_name
        # Store by just filename
        filename_to_class[os.path.basename(cls_name)] = cls_name
        # Store by parent/filename
        parts = cls_name.split('/')
        if len(parts) >= 2:
            filename_to_class['/'.join(parts[-2:])] = cls_name

    for sample in raw_samples:
        class_path = sample['class_path']

        # Try to find matching class
        matched_class = None

        # Direct match
        if class_path in class_to_idx:
            matched_class = class_path
        # By filename
        elif os.path.basename(class_path) in filename_to_class:
            matched_class = filename_to_class[os.path.basename(class_path)]
        # By parent/filename
        else:
            parts = class_path.split('/')
            if len(parts) >= 2:
                key = '/'.join(parts[-2:])
                if key in filename_to_class:
                    matched_class = filename_to_class[key]

        if matched_class:
            matched_samples.append({
                'image_path': sample['image_path'],
                'bbox': sample['bbox'],
                'class_idx': class_to_idx[matched_class],
                'class_name': matched_class
            })
        else:
            unmatched += 1

    print(f"Matched {len(matched_samples)} test samples")
    if unmatched > 0:
        print(f"WARNING: {unmatched} samples could not be matched!")
        
    return matched_samples


# =============================================================================
# Model Architecture (Exact Match with Notebook)
# =============================================================================
class ArcFaceHead(nn.Module):
    """ArcFace with margin annealing."""
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
        normalized_weights = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, normalized_weights)

        m = self.current_margin
        if m > 0:
            cos_m, sin_m = math.cos(m), math.sin(m)
            th = math.cos(math.pi - m)
            mm = math.sin(math.pi - m) * m

            sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
            phi = cosine * cos_m - sine * sin_m
            phi = torch.where(cosine > th, phi, cosine - mm)

            one_hot = F.one_hot(labels, self.num_classes).float()
            output = one_hot * phi + (1 - one_hot) * cosine
        else:
            output = cosine

        return output * self.scale

    def get_proxies(self):
        """Get L2-normalized proxies for HAL."""
        return F.normalize(self.weight, dim=1)


class HALHead(nn.Module):
    """
    Hierarchical Auxiliary Loss head.
    """
    def __init__(self, class_to_category, category_to_idx, scale=64.0):
        super().__init__()
        self.scale = scale
        self.num_categories = len(category_to_idx)

        # Build mapping: class_idx -> category_idx
        self.class_to_cat_idx = {}
        for class_name, cat_name in class_to_category.items():
            pass

        # Store which classes belong to which category
        self.category_to_classes = defaultdict(list)
        for class_name, cat_name in class_to_category.items():
            cat_idx = category_to_idx[cat_name]
            self.category_to_classes[cat_idx].append(class_name)

    def forward(self, embeddings, cat_labels, class_proxies, class_to_idx):
        return None  # Not needed for inference


class ProductRecognitionModel(nn.Module):
    def __init__(self, num_classes, num_categories, class_to_category, category_to_idx,
                 class_to_idx, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()

        # Backbone: ResNet-34
        resnet = models.resnet34(weights=None) # We will load weights
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
# Dataset Classes (Simplified for Inference)
# =============================================================================
class InferenceDataset(Dataset):
    """Simple dataset for inference (reference or test)."""
    
    def __init__(self, samples: List, transform=None, is_test=False):
        self.samples = samples
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        bbox = [0, 0, 0, 0]
        if self.is_test:
            sample = self.samples[idx]
            img_path = sample['image_path']
            class_idx = sample['class_idx']
            class_name = sample['class_name']
            
            # Handle bbox cropping
            try:
                image = Image.open(img_path).convert('RGB')
                x1, y1, x2, y2 = sample['bbox']
                bbox = [x1, y1, x2, y2]
                w, h = image.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
                
        else:
            # Training/Reference sample
            img_path, class_idx, class_name = self.samples[idx]
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx, class_name, img_path, torch.tensor(bbox)


# =============================================================================
# Evaluation Logic
# =============================================================================
def load_checkpoint(model: ProductRecognitionModel, checkpoint_path: Path):
    """Load checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Handle state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # The model keys should match exactly since we use the same class structure
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Warning: Strict load failed ({e}), trying strict=False")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(DEVICE)
    model.eval()
    return model


def build_reference_embeddings(
    model: ProductRecognitionModel,
    dataloader: DataLoader
) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    """Build reference embeddings from training data."""
    all_embeddings = []
    all_labels = []
    all_class_names = []
    all_paths = []
    
    print("Building reference database...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Unpack 5 items now
            images, labels, class_names, paths, _ = batch
            images = images.to(DEVICE)
            
            embeddings = model.get_embeddings(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.tolist())
            all_class_names.extend(class_names)
            all_paths.extend(paths)
    
    return np.vstack(all_embeddings), all_labels, all_class_names, all_paths


def evaluate_retrieval(
    model: ProductRecognitionModel,
    test_dataloader: DataLoader,
    ref_embeddings: np.ndarray,
    ref_labels: List[int],
    ref_class_names: List[str],
    ref_paths: List[str],
    top_k: int = 5
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate retrieval performance."""
    
    all_results = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    # Track per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Unpack 5 items
            images, labels, class_names, paths, bboxes = batch
            images = images.to(DEVICE)
            
            # Get query embeddings
            query_embeddings = model.get_embeddings(images).cpu().numpy()
            
            # Compute cosine similarities
            similarities = np.dot(query_embeddings, ref_embeddings.T)
            
            for i in range(len(images)):
                query_class = class_names[i]
                query_label = labels[i].item()
                query_path = paths[i]
                bbox = bboxes[i].tolist()
                
                # Get top-k matches
                top_k_indices = np.argsort(similarities[i])[::-1][:top_k]
                top_k_classes = [ref_class_names[idx] for idx in top_k_indices]
                top_k_scores = [similarities[i][idx] for idx in top_k_indices]
                top_k_paths = [ref_paths[idx] for idx in top_k_indices]
                
                class_total[query_class] += 1
                total += 1
                
                is_correct = False
                if top_k_classes[0] == query_class:
                    correct_top1 += 1
                    class_correct[query_class] += 1
                    is_correct = True
                
                if query_class in top_k_classes:
                    correct_top5 += 1
                
                # Store result
                all_results.append({
                    'query_path': query_path,
                    'bbox': bbox,
                    'query_class': query_class,
                    'top_k_classes': top_k_classes,
                    'top_k_scores': top_k_scores,
                    'top_k_paths': top_k_paths,
                    'is_correct': is_correct
                })
    
    # Metrics
    metrics = {
        'top1_accuracy': correct_top1 / total if total > 0 else 0,
        'top5_accuracy': correct_top5 / total if total > 0 else 0,
        'total_queries': total,
        'num_classes': len(class_total)
    }
    
    per_class_acc = {cls: class_correct[cls] / class_total[cls] for cls in class_total}
    metrics['per_class_accuracy'] = per_class_acc
    
    return metrics, all_results


def visualize_results(results: List[Dict], output_dir: Path, num_examples: int = NUM_EXAMPLE_IMAGES):
    """Visualize query vs top-5 predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample results
    correct = [r for r in results if r['is_correct']]
    incorrect = [r for r in results if not r['is_correct']]
    
    selected = []
    k_correct = min(len(correct), num_examples // 2)
    k_incorrect = min(len(incorrect), num_examples - k_correct)
    
    if correct: selected.extend(random.sample(correct, k_correct))
    if incorrect: selected.extend(random.sample(incorrect, k_incorrect))
    
    print(f"Generating {len(selected)} visualizations...")
    
    for idx, result in enumerate(selected):
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))
        
        # Query
        try:
            # Crop using the stored bbox
            img = Image.open(result['query_path']).convert('RGB')
            w, h = img.size
            x1, y1, x2, y2 = result['bbox']
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
            
            axes[0].imshow(img)
            axes[0].set_title(f"Query\n{result['query_class']}", fontsize=10)
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error: {e}", ha='center')
        axes[0].axis('off')
        
        for i, (path, cls, score) in enumerate(zip(result['top_k_paths'], result['top_k_classes'], result['top_k_scores'])):
            ax = axes[i+1]
            try:
                img = Image.open(path).convert('RGB')
                ax.imshow(img)
                color = 'green' if cls == result['query_class'] else 'red'
                ax.set_title(f"Top {i+1}: {cls}\n{score:.3f}", fontsize=9, color=color)
            except:
                ax.text(0.5, 0.5, "Error", ha='center')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_dir / f"viz_{idx}.png", bbox_inches='tight')
        plt.close()


def main():
    print("=" * 60)
    print("ArcFace Evaluation (Fixed Version)")
    print("=" * 60)
    
    # 1. Parse Annotations
    print("\nStep 1: Parsing Annotations...")
    test_samples_raw, annotation_classes = parse_all_annotations(ANNOTATIONS_DIR, TESTING_DIR)
    
    if not annotation_classes:
        print("No valid annotations found. Exiting.")
        return

    # 2. Build Training Set (Reference DB)
    print("\nStep 2: Building Reference Database Indices...")
    class_to_idx, idx_to_class, class_to_category, category_to_idx, image_paths = build_training_set(
        TRAINING_DIR, valid_classes=annotation_classes
    )
    
    # 3. Build Test Set
    print("\nStep 3: Matching Test Samples...")
    test_samples = build_test_samples(test_samples_raw, class_to_idx)
    
    # 4. Initialize Model
    print("\nStep 4: Loading Model...")
    model = ProductRecognitionModel(
        num_classes=len(class_to_idx),
        num_categories=len(category_to_idx),
        class_to_category=class_to_category,
        category_to_idx=category_to_idx,
        class_to_idx=class_to_idx,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)
    
    checkpoint_path = CHECKPOINT_DIR / "best_2.pth"
    try:
        model = load_checkpoint(model, checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Try finding any .pth
        pths = list(CHECKPOINT_DIR.glob("*.pth"))
        if pths:
            print(f"Trying alternative: {pths[0]}")
            model = load_checkpoint(model, pths[0])
        else:
            print("No checkpoints found!")
            return

    # 5. Create Dataloaders
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ref_dataset = InferenceDataset(image_paths, transform=test_transform, is_test=False)
    test_dataset = InferenceDataset(test_samples, transform=test_transform, is_test=True)
    
    ref_loader = DataLoader(ref_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load/Build reference database
    ref_db_path = OUTPUT_DIR / "reference_db.pt"
    if ref_db_path.exists():
        print(f"\nLoading reference database from {ref_db_path}...")
        ref_data = torch.load(ref_db_path, weights_only=False)
        ref_embeddings = ref_data['embeddings']
        ref_labels = ref_data['labels']
        ref_class_names = ref_data['class_names']
        ref_paths = ref_data['paths']
        print(f"Loaded {len(ref_embeddings)} reference embeddings.")
    else:
        print("\nBuilding reference database...")
        ref_embeddings, ref_labels, ref_class_names, ref_paths = build_reference_embeddings(
            model, ref_loader
        )
        print(f"Saving reference database to {ref_db_path}...")
        torch.save({
            'embeddings': ref_embeddings,
            'labels': ref_labels,
            'class_names': ref_class_names,
            'paths': ref_paths
        }, ref_db_path)
    
    # 6. Evaluate
    metrics, results = evaluate_retrieval(
        model, test_loader, ref_embeddings, ref_labels, ref_class_names, ref_paths, top_k=TOP_K
    )
    
    # 7. Print & Save
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Top-1 Accuracy: {metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}%")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metrics_fixed.txt", "w") as f:
        f.write(f"Top-1: {metrics['top1_accuracy']:.4f}\n")
        f.write(f"Top-5: {metrics['top5_accuracy']:.4f}\n")
        
    visualize_results(results, OUTPUT_DIR)
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
