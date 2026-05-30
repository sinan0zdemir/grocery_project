"""
ArcFace v4 + Live OCR Reranking Evaluation

Evaluates the ResNet-34 + ArcFace + HAL model with live EasyOCR on each crop.
When OCR finds confident text that fuzzy-matches a product name, it boosts
that product's KNN candidates. Falls back to pure visual when OCR finds nothing.

Tracks baseline vs OCR-boosted metrics side-by-side with confusion analysis.

Usage:
    python evaluate_arcface_v4_ocr.py
    python evaluate_arcface_v4_ocr.py --checkpoint "best _newest.pth" --ocr_weight 0.15
"""

import os
import sys
import re
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from difflib import SequenceMatcher

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

CHECKPOINT_DIR = SCRIPT_DIR.parent / "checkpoints"
DEFAULT_CHECKPOINT = "best_newest.pth"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "arcface_v4_ocr"

FEATURE_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 20
OCR_WEIGHT = 0.7
OCR_CONF = 0.25
FUZZY_THRESHOLD = 0.50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading (same as v4 eval)
# =============================================================================
def load_product_id_mapping(annotations_dir):
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
                    id_to_name[int(parts[0])] = parts[1].strip()
                except ValueError:
                    continue
    return id_to_name


def normalize_product_name(name):
    return os.path.splitext(name)[0].lower().strip()


def parse_annotations(annotations_dir, testing_dir, id_to_name):
    df = pd.read_csv(annotations_dir / "_annotations.csv")
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
    print(f"  {len(samples)} test samples from {df['filename'].nunique()} images")
    return samples


def build_training_set(training_dir, id_to_name, valid_product_ids=None):
    name_to_id = {}
    for pid, pname in id_to_name.items():
        name_to_id[normalize_product_name(pname)] = pid

    image_paths, class_to_idx, idx_to_class = [], {}, {}
    all_images = []
    for root, dirs, files in os.walk(str(training_dir)):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append((os.path.join(root, f), f))

    for full_path, filename in all_images:
        norm = normalize_product_name(filename)
        product_id = None
        for norm_name, pid in name_to_id.items():
            if norm_name == norm or norm_name in norm or norm in norm_name:
                product_id = pid
                break
        if product_id is None:
            continue
        if valid_product_ids and product_id not in valid_product_ids:
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
    matched, unmatched = [], set()
    for s in raw_samples:
        class_name = id_to_name.get(s["product_id"], "")
        if class_name in class_to_idx:
            matched.append({
                "image_path": s["image_path"], "bbox": s["bbox"],
                "class_idx": class_to_idx[class_name], "class_name": class_name,
            })
        else:
            unmatched.add(class_name)
    if unmatched:
        print(f"  WARNING: {len(unmatched)} unmatched products")
    print(f"  {len(matched)} matched test samples")
    return matched


# =============================================================================
# OCR Engine — PP-OCRv5 Mobile (PaddleOCR 3.x)
# =============================================================================
def init_ocr():
    print("  Initializing PP-OCRv5 (mobile)...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print(f"  [OK] PP-OCRv5 mobile ready")
    return ocr


def run_ocr(reader, crop_pil, min_conf):
    try:
        img_array = np.array(crop_pil)
        result = reader.predict(img_array)
        fragments = []
        if result and len(result) > 0:
            item = result[0]
            texts = item.get('rec_texts', []) if hasattr(item, 'get') else []
            scores = item.get('rec_scores', []) if hasattr(item, 'get') else []
            for text, score in zip(texts, scores):
                t = text.strip()
                if len(t) >= 2 and float(score) >= min_conf:
                    fragments.append((t, float(score)))
        return fragments
    except Exception as e:
        return []



# =============================================================================
# Fuzzy Text Matching
# =============================================================================
def clean(text):
    text = re.sub(r'[^a-zA-Z\u00e7\u00c7\u011f\u011e\u0131\u0130\u00f6\u00d6\u015f\u015e\u00fc\u00dc0-9\s]', '', text)
    return text.lower().strip()


def char_ngrams(text, n=3):
    t = clean(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def fuzzy_match_product(ocr_fragments, product_names):
    """
    Match OCR text against product names. Returns {name: score} for matches
    above FUZZY_THRESHOLD. Uses token match, substring, trigrams, SequenceMatcher.
    """
    scores = {}
    for product_name in product_names:
        pname_clean = clean(product_name)
        pname_words = [w for w in pname_clean.split() if len(w) >= 2]
        if not pname_words:
            continue

        best_score = 0.0
        for ocr_text, ocr_conf in ocr_fragments:
            ocr_clean = clean(ocr_text)
            if len(ocr_clean) < 2:
                continue

            for pw in pname_words:
                # Exact token
                if ocr_clean == pw:
                    best_score = max(best_score, 1.0 * ocr_conf)
                    continue

                # Substring (min 3 chars)
                if len(ocr_clean) >= 3 and len(pw) >= 3:
                    if ocr_clean in pw:
                        best_score = max(best_score, (len(ocr_clean) / len(pw)) * ocr_conf)
                    elif pw in ocr_clean:
                        best_score = max(best_score, (len(pw) / len(ocr_clean)) * ocr_conf)

                # Trigram overlap
                if len(ocr_clean) >= 3 and len(pw) >= 3:
                    og, pg = char_ngrams(ocr_clean), char_ngrams(pw)
                    if og and pg:
                        overlap = len(og & pg) / max(len(og), len(pg))
                        if overlap > 0.4:
                            best_score = max(best_score, overlap * 0.85 * ocr_conf)

                # SequenceMatcher
                ratio = SequenceMatcher(None, ocr_clean, pw).ratio()
                if ratio > 0.6:
                    best_score = max(best_score, ratio * 0.9 * ocr_conf)

        if best_score >= FUZZY_THRESHOLD:
            scores[product_name] = best_score

    return scores


# =============================================================================
# Model (same as v4)
# =============================================================================
class ProductRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, scale=64.0, margin=0.5):
        super().__init__()
        resnet = models.resnet34(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim), nn.BatchNorm1d(embedding_dim)
        )
        self.arcface = ArcFaceHead(embedding_dim, num_classes, scale, margin)
        self.embedding_dim = embedding_dim

    def get_embeddings(self, x):
        return F.normalize(self.embedding(self.backbone(x).flatten(1)), dim=1)

    def forward(self, x):
        return self.get_embeddings(x)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin, self.current_margin = scale, margin, margin
        self.num_classes = num_classes

    def set_margin(self, m): self.current_margin = m
    def forward(self, e, l=None): return self.scale * F.linear(F.normalize(e), F.normalize(self.weight))
    def get_proxies(self): return F.normalize(self.weight, dim=1)


class AutoCropWhiteBackground:
    def __init__(self, threshold=240, padding=20, max_size=(1400, 700)):
        self.threshold, self.padding, self.max_size = threshold, padding, max_size

    def __call__(self, img):
        arr = np.array(img)
        nw = ~np.all(arr > self.threshold, axis=2)
        rows, cols = np.any(nw, axis=1), np.any(nw, axis=0)
        if not np.any(rows) or not np.any(cols):
            return transforms.CenterCrop(self.max_size)(img)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        h, w = arr.shape[:2]
        y0, y1 = max(0, y0-self.padding), min(h, y1+self.padding)
        x0, x1 = max(0, x0-self.padding), min(w, x1+self.padding)
        cropped = img.crop((x0, y0, x1, y1))
        cw, ch = cropped.size
        mh, mw = self.max_size
        if ch > mh or cw > mw:
            cropped = transforms.CenterCrop(self.max_size)(cropped)
        return cropped


class TestDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform, self._cache = samples, transform, {}

    def __len__(self): return len(self.samples)

    def get_crop_pil(self, idx):
        """Get raw PIL crop for OCR (no transform)."""
        s = self.samples[idx]
        try:
            if s["image_path"] not in self._cache:
                self._cache[s["image_path"]] = Image.open(s["image_path"]).convert("RGB")
            img = self._cache[s["image_path"]]
            x1, y1, x2, y2 = s["bbox"]
            w, h = img.size
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
            return img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else None
        except Exception:
            return None

    def __getitem__(self, idx):
        crop = self.get_crop_pil(idx)
        if crop is None:
            crop = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(crop), self.samples[idx]["class_idx"], idx


# =============================================================================
# Reference DB
# =============================================================================
@torch.no_grad()
def build_reference_db(model, image_paths, transform, device):
    model.eval()
    embeddings, labels = [], []
    for path, class_idx, _ in tqdm(image_paths, desc="Ref DB"):
        try:
            img = Image.open(path).convert("RGB")
            emb = model.get_embeddings(transform(img).unsqueeze(0).to(device)).cpu().numpy()
            embeddings.append(emb)
            labels.append(class_idx)
        except Exception:
            pass
    return np.vstack(embeddings), np.array(labels)


# =============================================================================
# Evaluation with OCR Reranking
# =============================================================================
@torch.no_grad()
def evaluate_with_ocr(
    model, dataloader, dataset, ref_emb, ref_lbl, device,
    idx_to_class, ocr_reader, unique_names,
    top_k, ocr_weight, ocr_conf,
):
    model.eval()
    total = 0
    # Baseline counters
    base_hr = {1: 0, 5: 0, 10: 0, 20: 0}
    # OCR-boosted counters
    ocr_hr = {1: 0, 5: 0, 10: 0, 20: 0}

    base_correct, base_total = defaultdict(int), defaultdict(int)
    ocr_correct_map, ocr_total_map = defaultdict(int), defaultdict(int)

    base_confusion = defaultdict(int)
    ocr_confusion = defaultdict(int)

    ocr_used_count = 0
    ocr_helped = 0
    ocr_hurt = 0
    ocr_time = 0.0
    all_results = []

    for batch_imgs, batch_lbls, batch_idxs in tqdm(dataloader, desc="Eval"):
        batch_imgs = batch_imgs.to(device)
        embs = model.get_embeddings(batch_imgs).cpu().numpy()
        sims = embs @ ref_emb.T

        for i in range(len(batch_imgs)):
            true_lbl = batch_lbls[i].item()
            sample_idx = batch_idxs[i].item()
            true_name = idx_to_class.get(true_lbl, f"?{true_lbl}")
            total += 1
            base_total[true_lbl] += 1
            ocr_total_map[true_lbl] += 1

            # --- Top-K visual retrieval ---
            sorted_idx = np.argsort(-sims[i])[:top_k]
            top_classes = ref_lbl[sorted_idx]
            top_scores = sims[i][sorted_idx]
            top_names = [idx_to_class.get(int(c), f"?{c}") for c in top_classes]

            # --- Baseline metrics ---
            base_pred = int(top_classes[0])
            if base_pred == true_lbl:
                base_correct[true_lbl] += 1
            else:
                base_confusion[(true_lbl, base_pred)] += 1

            for k in [1, 5, 10, 20]:
                if k <= top_k and true_lbl in top_classes[:k]:
                    base_hr[k] += 1

            # --- Live OCR ---
            crop_pil = dataset.get_crop_pil(sample_idx)
            ocr_fragments = []
            ocr_matches = {}
            if crop_pil is not None:
                t0 = time.time()
                ocr_fragments = run_ocr(ocr_reader, crop_pil, ocr_conf)
                ocr_time += time.time() - t0

                if ocr_fragments:
                    ocr_matches = fuzzy_match_product(ocr_fragments, unique_names)

            used_ocr = bool(ocr_matches)
            if used_ocr:
                ocr_used_count += 1

            # --- Rerank with OCR: combined = visual + weight * fuzzy_score ---
            boosted = []
            for k_idx in range(len(top_classes)):
                vis = float(top_scores[k_idx])
                cname = top_names[k_idx]
                bonus = ocr_matches.get(cname, 0.0)
                boosted.append((int(top_classes[k_idx]), cname, vis, bonus, vis + ocr_weight * bonus))

            boosted.sort(key=lambda x: x[4], reverse=True)

            # --- OCR-boosted metrics ---
            ocr_pred = boosted[0][0]
            if ocr_pred == true_lbl:
                ocr_correct_map[true_lbl] += 1
            else:
                ocr_confusion[(true_lbl, ocr_pred)] += 1

            boosted_classes = [b[0] for b in boosted]
            for k in [1, 5, 10, 20]:
                if k <= top_k and true_lbl in boosted_classes[:k]:
                    ocr_hr[k] += 1

            # Track helped/hurt
            base_ok = (base_pred == true_lbl)
            ocr_ok = (ocr_pred == true_lbl)
            if not base_ok and ocr_ok: ocr_helped += 1
            if base_ok and not ocr_ok: ocr_hurt += 1

            ocr_text = " | ".join(f"{t}({c:.2f})" for t, c in ocr_fragments) if ocr_fragments else ""
            match_text = ", ".join(f"{safe(n)}={s:.2f}" for n, s in ocr_matches.items()) if ocr_matches else ""

            all_results.append({
                "true_lbl": true_lbl, "true_name": true_name,
                "base_pred": base_pred, "ocr_pred": ocr_pred,
                "base_ok": base_ok, "ocr_ok": ocr_ok,
                "ocr_text": ocr_text, "ocr_matches": match_text,
                "used_ocr": used_ocr, "boosted": boosted[:5],
                # Raw data for ablation sweep
                "top_classes": [int(c) for c in top_classes],
                "top_scores": [float(s) for s in top_scores],
                "ocr_match_scores": dict(ocr_matches),  # {name: fuzzy_score}
                "top_names": list(top_names),
            })

    return {
        "total": total,
        "base_hr": {k: v/total*100 for k, v in base_hr.items()},
        "ocr_hr": {k: v/total*100 for k, v in ocr_hr.items()},
        "base_correct": dict(base_correct), "base_total": dict(base_total),
        "ocr_correct": dict(ocr_correct_map), "ocr_total": dict(ocr_total_map),
        "base_confusion": sorted(base_confusion.items(), key=lambda x: -x[1]),
        "ocr_confusion": sorted(ocr_confusion.items(), key=lambda x: -x[1]),
        "ocr_used": ocr_used_count,
        "ocr_helped": ocr_helped, "ocr_hurt": ocr_hurt,
        "ocr_avg_ms": ocr_time / total * 1000 if total else 0,
    }, all_results


# =============================================================================
# Ablation Sweep (post-hoc, no re-running OCR)
# =============================================================================
def ablation_sweep(all_results, weights, top_k=20):
    """
    Re-rank cached results at each OCR weight. Returns list of dicts:
    [{weight, hr1, hr5, helped, hurt}, ...]
    """
    sweep = []
    for w in weights:
        hr = {1: 0, 5: 0, 10: 0, 20: 0}
        helped, hurt = 0, 0
        total = len(all_results)

        for r in all_results:
            true_lbl = r["true_lbl"]
            top_classes = r["top_classes"]
            top_scores = r["top_scores"]
            top_names = r["top_names"]
            ocr_scores = r["ocr_match_scores"]

            # Rerank
            boosted = []
            for idx in range(len(top_classes)):
                vis = top_scores[idx]
                bonus = ocr_scores.get(top_names[idx], 0.0)
                boosted.append((top_classes[idx], vis + w * bonus))
            boosted.sort(key=lambda x: x[1], reverse=True)
            boosted_classes = [b[0] for b in boosted]

            for k in [1, 5, 10, 20]:
                if k <= top_k and true_lbl in boosted_classes[:k]:
                    hr[k] += 1

            # Base is w=0 prediction (first in top_classes)
            base_ok = (top_classes[0] == true_lbl)
            ocr_ok = (boosted_classes[0] == true_lbl)
            if not base_ok and ocr_ok: helped += 1
            if base_ok and not ocr_ok: hurt += 1

        sweep.append({
            "weight": w,
            "hr1": hr[1] / total * 100,
            "hr5": hr[5] / total * 100,
            "hr10": hr[10] / total * 100 if 10 <= top_k else 0,
            "helped": helped, "hurt": hurt,
            "net": helped - hurt,
        })
    return sweep


def plot_ablation(sweep, baseline_hr1, output_path):
    """Two-panel ablation plot: accuracy curve + net gain bars."""
    weights = [s["weight"] for s in sweep]
    hr1s = [s["hr1"] for s in sweep]
    nets = [s["net"] for s in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Accuracy vs weight
    ax1.plot(weights, hr1s, 'b-o', label='OCR Reranked', markersize=6)
    ax1.axhline(y=baseline_hr1, color='r', linestyle='--', label='Baseline')
    ax1.set_xlabel('OCR Weight')
    ax1.set_ylabel('Top-1 Name Accuracy (%)')
    ax1.set_title('Accuracy vs OCR Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Net gain bar chart
    colors = ['green' if n >= 0 else 'red' for n in nets]
    ax2.bar(weights, nets, width=0.04, color=colors, alpha=0.7)
    ax2.set_xlabel('OCR Weight')
    ax2.set_ylabel('Net Gain (helped - hurt)')
    ax2.set_title('OCR Impact per Weight')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ablation plot: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--ocr_weight", type=float, default=OCR_WEIGHT)
    parser.add_argument("--ocr_conf", type=float, default=OCR_CONF)
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_DIR / args.checkpoint

    print("=" * 70)
    print("ArcFace v4 + Live OCR Reranking - Migros v6")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"OCR weight: {args.ocr_weight}, OCR conf: {args.ocr_conf}")
    print()

    for p, n in [(TRAINING_DIR, "Training"), (TESTING_DIR, "Testing"),
                  (ANNOTATIONS_DIR / "_annotations.csv", "Annotations"),
                  (checkpoint_path, "Checkpoint")]:
        if not p.exists():
            print(f"  ERROR: {n} not found: {p}")
            return
        print(f"  [OK] {n}")

    # 1. Load checkpoint first
    print(f"\n1. Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if "class_to_idx" not in ckpt:
        print("  ERROR: No class_to_idx in checkpoint!")
        return

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = ckpt.get("idx_to_class", {v: k for k, v in class_to_idx.items()})
    NUM_CLASSES = ckpt.get("num_classes", len(class_to_idx))
    id_to_name = ckpt.get("id_to_name", load_product_id_mapping(ANNOTATIONS_DIR))
    print(f"  {NUM_CLASSES} classes, epoch {ckpt.get('epoch', '?')}")

    unique_names = sorted(set(idx_to_class.values()))
    print(f"  {len(unique_names)} unique product names")

    # 2. Build model
    print(f"\n2. Loading model...")
    model = ProductRecognitionModel(NUM_CLASSES, FEATURE_DIM).to(DEVICE)
    sd = ckpt["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"  Loaded ({len(filtered)} keys)")

    # 3. Data
    print(f"\n3. Loading data...")
    raw_samples = parse_annotations(ANNOTATIONS_DIR, TESTING_DIR, id_to_name)
    test_samples = build_test_samples(raw_samples, class_to_idx, id_to_name)

    annotation_pids = set(s["product_id"] for s in raw_samples)
    _, _, image_paths_raw = build_training_set(TRAINING_DIR, id_to_name, annotation_pids)
    image_paths = [(p, class_to_idx[n], n) for p, _, n in image_paths_raw if n in class_to_idx]
    print(f"  {len(image_paths)} training images remapped")

    # 4. Transforms
    ref_transform = transforms.Compose([
        AutoCropWhiteBackground(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # 5. Reference DB
    print(f"\n4. Building reference DB...")
    t0 = time.time()
    ref_emb, ref_lbl = build_reference_db(model, image_paths, ref_transform, DEVICE)
    print(f"  {len(ref_emb)} embeddings ({time.time()-t0:.1f}s)")

    # 6. OCR
    print(f"\n5. Initializing OCR...")
    ocr_reader = init_ocr()

    # 7. Evaluate
    dataset = TestDataset(test_samples, test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n6. Evaluating ({len(test_samples)} crops)...")
    t0 = time.time()
    metrics, results = evaluate_with_ocr(
        model, loader, dataset, ref_emb, ref_lbl, DEVICE,
        idx_to_class, ocr_reader, unique_names,
        args.top_k, args.ocr_weight, args.ocr_conf,
    )
    eval_time = time.time() - t0

    # 8. Print results
    m = metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    print(f"  Test crops: {m['total']},  Ref DB: {len(ref_emb)} embeddings")
    print(f"  Eval time: {eval_time:.1f}s  (OCR avg: {m['ocr_avg_ms']:.0f}ms/crop)")
    print(f"  OCR found match: {m['ocr_used']}/{m['total']} ({m['ocr_used']*100//m['total']}%)")
    print()

    print("  +-------------------+-----------+-----------+--------+")
    print("  | Metric            |  Baseline |  OCR+KNN  |  Delta |")
    print("  +-------------------+-----------+-----------+--------+")
    for k in [1, 5, 10, 20]:
        if k > args.top_k: break
        bv, ov = m["base_hr"][k], m["ocr_hr"][k]
        d = ov - bv
        print(f"  | HR@{k:<15} | {bv:8.2f}% | {ov:8.2f}% | {d:+5.2f}% |")
    print("  +-------------------+-----------+-----------+--------+")
    print(f"\n  OCR helped: {m['ocr_helped']},  OCR hurt: {m['ocr_hurt']},  net: {m['ocr_helped']-m['ocr_hurt']:+d}")

    # Confusion — baseline
    print(f"\n  TOP CONFUSED PAIRS (baseline):")
    print("  " + "-" * 65)
    for i, ((t, p), c) in enumerate(m["base_confusion"][:10]):
        tn, pn = safe(idx_to_class.get(t, "?")), safe(idx_to_class.get(p, "?"))
        print(f"  {i+1:2d}. {tn[:35]:<35} -> {pn[:25]} ({c})")

    # Confusion — OCR-boosted
    if m["ocr_confusion"] != m["base_confusion"]:
        print(f"\n  TOP CONFUSED PAIRS (OCR-boosted):")
        print("  " + "-" * 65)
        for i, ((t, p), c) in enumerate(m["ocr_confusion"][:10]):
            tn, pn = safe(idx_to_class.get(t, "?")), safe(idx_to_class.get(p, "?"))
            print(f"  {i+1:2d}. {tn[:35]:<35} -> {pn[:25]} ({c})")

    # Per-class comparison
    print(f"\n  PER-CLASS: Baseline vs OCR (classes with change):")
    print("  " + "-" * 70)
    print(f"  {'Class':<40} {'Base':>8} {'OCR':>8} {'Diff':>6}")
    for cls_idx in sorted(m["base_total"].keys()):
        tot = m["base_total"][cls_idx]
        b = m["base_correct"].get(cls_idx, 0)
        o = m["ocr_correct"].get(cls_idx, 0)
        if b != o:
            name = safe(idx_to_class.get(cls_idx, "?"))[:40]
            print(f"  {name:<40} {b:>3}/{tot:<3}  {o:>3}/{tot:<3}  {o-b:>+4}")

    # Show OCR examples where it helped
    helped = [r for r in results if not r["base_ok"] and r["ocr_ok"]]
    if helped:
        print(f"\n  OCR HELPED ({len(helped)} cases):")
        print("  " + "-" * 65)
        for r in helped[:15]:
            print(f"  GT: {safe(r['true_name'][:35]):<35} OCR: {safe(r['ocr_text'][:50])}")
            print(f"    Matches: {safe(r['ocr_matches'][:70])}")
            print()

    # Show OCR examples where it hurt
    hurt = [r for r in results if r["base_ok"] and not r["ocr_ok"]]
    if hurt:
        print(f"\n  OCR HURT ({len(hurt)} cases):")
        print("  " + "-" * 65)
        for r in hurt[:10]:
            print(f"  GT: {safe(r['true_name'][:35]):<35} OCR: {safe(r['ocr_text'][:50])}")
            print(f"    Matches: {safe(r['ocr_matches'][:70])}")
            pred_name = safe(idx_to_class.get(r['ocr_pred'], '?'))
            print(f"    Wrong pred: {pred_name}")
            print()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write(f"ArcFace v4 + Live OCR - Migros v6\n{'='*50}\n")
        f.write(f"Checkpoint: {args.checkpoint}, Epoch: {ckpt.get('epoch','?')}\n")
        f.write(f"OCR weight: {args.ocr_weight}, conf: {args.ocr_conf}\n")
        f.write(f"Total: {m['total']}, OCR used: {m['ocr_used']}\n\n")
        for k in [1, 5, 10, 20]:
            if k > args.top_k: break
            f.write(f"HR@{k} Baseline: {m['base_hr'][k]:.2f}%  OCR: {m['ocr_hr'][k]:.2f}%  "
                    f"Delta: {m['ocr_hr'][k]-m['base_hr'][k]:+.2f}%\n")
        f.write(f"\nHelped: {m['ocr_helped']}, Hurt: {m['ocr_hurt']}\n")

    import csv
    with open(OUTPUT_DIR / "per_crop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_name","base_pred","ocr_pred","base_ok","ocr_ok","ocr_text","ocr_matches","used_ocr"])
        for r in results:
            bp = idx_to_class.get(r["base_pred"], "?")
            op = idx_to_class.get(r["ocr_pred"], "?")
            w.writerow([r["true_name"], bp, op, r["base_ok"], r["ocr_ok"],
                        r["ocr_text"], r["ocr_matches"], r["used_ocr"]])

    # Ablation sweep
    print(f"\n7. Running ablation sweep...")
    sweep_weights = [round(w * 0.05, 2) for w in range(0, 15)]  # 0.0 to 0.70
    sweep = ablation_sweep(results, sweep_weights, args.top_k)
    baseline_hr1 = m["base_hr"][1]

    print("\n  +--------+--------+--------+--------+---------+")
    print("  | Weight |  HR@1  |  HR@5  | Helped |   Hurt  |")
    print("  +--------+--------+--------+--------+---------+")
    for s in sweep:
        print(f"  | {s['weight']:5.2f}  | {s['hr1']:5.2f}% | {s['hr5']:5.2f}% | {s['helped']:6d} | {s['hurt']:7d} |")
    print("  +--------+--------+--------+--------+---------+")

    best = max(sweep, key=lambda s: s["hr1"])
    print(f"\n  Best weight: {best['weight']:.2f} -> HR@1={best['hr1']:.2f}% (net={best['net']:+d})")

    plot_ablation(sweep, baseline_hr1, OUTPUT_DIR / "ablation_curve.png")

    # Save ablation data
    with open(OUTPUT_DIR / "ablation_sweep.txt", "w") as f:
        f.write("weight,hr1,hr5,helped,hurt,net\n")
        for s in sweep:
            f.write(f"{s['weight']:.2f},{s['hr1']:.2f},{s['hr5']:.2f},{s['helped']},{s['hurt']},{s['net']}\n")

    print(f"\n  Results: {OUTPUT_DIR / 'results.txt'}")
    print(f"  Per-crop: {OUTPUT_DIR / 'per_crop.csv'}")
    print(f"  Ablation: {OUTPUT_DIR / 'ablation_sweep.txt'}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
