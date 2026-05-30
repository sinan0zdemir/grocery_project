"""
ArcFace + Live OCR Reranking Evaluation on Migros v6

Runs PP-OCRv5 (mobile) on each crop at evaluation time. When OCR finds
confident text that fuzzy-matches a product name, it boosts that product's
candidates in the KNN retrieval. When OCR finds nothing useful, it falls
back to pure visual retrieval (no penalty).

Tracks both ID-level and name-level accuracy side-by-side.

Usage:
    python evaluate_ocr_rerank.py
    python evaluate_ocr_rerank.py --top_k 10 --ocr_weight 0.15 --ocr_conf 0.3
"""

import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
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
_TR = str.maketrans('ccGgIiOoSsUu', 'ccGgIiOoSsUu')
_TR_MAP = str.maketrans('\u00e7\u00c7\u011f\u011e\u0131\u0130\u00f6\u00d6\u015f\u015e\u00fc\u00dc', 'cCgGiIoOsSuU')
def safe(s): return str(s).translate(_TR_MAP)

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
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "ocr_rerank"

EMBEDDING_DIM = 512
IMG_SIZE = 224
BATCH_SIZE = 32
TOP_K = 10
OCR_WEIGHT = 0.15        # Additive bonus weight for OCR matches
OCR_CONF_THRESHOLD = 0.3  # Min OCR confidence to consider a text fragment
FUZZY_THRESHOLD = 0.55    # Min fuzzy score to count as a match
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Loading
# =============================================================================
def load_product_mapping(csv_path: Path) -> Dict[str, str]:
    mapping = {}
    if not csv_path.exists():
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
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        filename = row["filename"]
        class_id = str(int(row["class"]))
        bbox = [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])]
        img_path = testing_dir / filename
        if not img_path.exists():
            continue
        samples.append({
            "image_path": str(img_path),
            "class_id": class_id,
            "bbox": bbox,
            "filename": filename
        })
    print(f"Loaded {len(samples)} annotated crops from {df['filename'].nunique()} images")
    return samples


# =============================================================================
# OCR Engine — PP-OCRv5 Mobile (PaddleOCR 3.x)
# =============================================================================
def init_ocr():
    """Initialize PP-OCRv5 mobile reader."""
    print("Initializing PP-OCRv5 (mobile)...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print(f"  [OK] PP-OCRv5 mobile ready")
    return ocr


def run_ocr(reader, crop_pil: Image.Image, min_conf: float) -> List[Tuple[str, float]]:
    """
    Run OCR on a crop. Returns list of (text, confidence) tuples
    filtered by min_conf. Returns empty list if nothing found.
    """
    try:
        crop_np = np.array(crop_pil)
        result = reader.predict(crop_np)
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
    except Exception:
        return []


# =============================================================================
# Fuzzy Text Matching
# =============================================================================
def clean(text: str) -> str:
    """Normalize text for matching: lowercase, remove special chars."""
    text = re.sub(r'[^a-zA-Z\u00e7\u00c7\u011f\u011e\u0131\u0130\u00f6\u00d6\u015f\u015e\u00fc\u00dc0-9\s]', '', text)
    return text.lower().strip()


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    t = clean(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i+n] for i in range(len(t) - n + 1)}


def fuzzy_match_product(
    ocr_fragments: List[Tuple[str, float]],
    product_names: List[str],
) -> Dict[str, float]:
    """
    Given OCR fragments, find fuzzy matches against all known product names.
    Returns {product_name: best_match_score} for matches above FUZZY_THRESHOLD.

    Match signals (all dynamic, no hardcoded rules):
    - Exact token match: "PEPSI" in OCR matches "Pepsi" in product name
    - Substring: "arbell" found inside "Arbella"
    - Trigram overlap: handles OCR noise like "ARPEILA" ~ "ARBELLA"
    - SequenceMatcher: overall fuzzy similarity
    """
    scores = {}

    for product_name in product_names:
        pname_clean = clean(product_name)
        pname_words = [w for w in pname_clean.split() if len(w) >= 2]
        if not pname_words:
            continue

        best_score = 0.0
        matched_words = 0

        for ocr_text, ocr_conf in ocr_fragments:
            ocr_clean = clean(ocr_text)
            if len(ocr_clean) < 2:
                continue

            # Check each OCR token against each product word
            for pw in pname_words:
                # 1. Exact match
                if ocr_clean == pw:
                    best_score = max(best_score, 1.0 * ocr_conf)
                    matched_words += 1
                    continue

                # 2. Substring containment (min 3 chars)
                if len(ocr_clean) >= 3 and len(pw) >= 3:
                    if ocr_clean in pw:
                        s = (len(ocr_clean) / len(pw)) * ocr_conf
                        best_score = max(best_score, s)
                    elif pw in ocr_clean:
                        s = (len(pw) / len(ocr_clean)) * ocr_conf
                        best_score = max(best_score, s)

                # 3. Trigram overlap
                if len(ocr_clean) >= 3 and len(pw) >= 3:
                    og = char_ngrams(ocr_clean, 3)
                    pg = char_ngrams(pw, 3)
                    if og and pg:
                        overlap = len(og & pg) / max(len(og), len(pg))
                        if overlap > 0.4:
                            best_score = max(best_score, overlap * 0.85 * ocr_conf)

                # 4. SequenceMatcher
                ratio = SequenceMatcher(None, ocr_clean, pw).ratio()
                if ratio > 0.6:
                    best_score = max(best_score, ratio * 0.9 * ocr_conf)

        if best_score >= FUZZY_THRESHOLD:
            scores[product_name] = best_score

    return scores


# =============================================================================
# Model
# =============================================================================
def load_model(checkpoint_path: Path) -> nn.Module:
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, EMBEDDING_DIM)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(state_dict, strict=True)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    return backbone


def load_reference_db(db_path: Path):
    data = torch.load(db_path, map_location="cpu", weights_only=False)
    return data["embeddings"], data["class_names"], data["labels"]


# =============================================================================
# Dataset
# =============================================================================
class CropDataset(Dataset):
    """Returns both the transformed tensor AND the raw PIL crop for OCR."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def _get_crop(self, idx):
        s = self.samples[idx]
        img_path = s["image_path"]
        if img_path not in self._cache:
            self._cache[img_path] = Image.open(img_path).convert("RGB")
        img = self._cache[img_path]
        x1, y1, x2, y2 = s["bbox"]
        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            return img.crop((x1, y1, x2, y2))
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

    def __getitem__(self, idx):
        crop = self._get_crop(idx)
        tensor = self.transform(crop)
        return tensor, self.samples[idx]["class_id"], idx


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(
    model, dataloader, dataset, ref_embeddings, ref_class_names,
    samples, product_map, ocr_reader,
    unique_product_names, top_k, ocr_weight, ocr_conf,
):
    all_results = []

    # Counters: baseline (visual only)
    base_id = {1: 0, 5: 0, 10: 0}
    base_name = {1: 0, 5: 0, 10: 0}
    # Counters: OCR-boosted
    ocr_id = {1: 0, 5: 0, 10: 0}
    ocr_name = {1: 0, 5: 0, 10: 0}

    total = 0
    ocr_used = 0
    ocr_helped_id = 0
    ocr_helped_name = 0
    ocr_hurt_id = 0
    ocr_hurt_name = 0
    ocr_time_total = 0.0

    per_class_base = defaultdict(int)
    per_class_ocr = defaultdict(int)
    per_class_total = defaultdict(int)

    print("Evaluating...")
    with torch.no_grad():
        for batch_imgs, batch_class_ids, batch_indices in tqdm(dataloader, desc="Eval"):
            batch_imgs = batch_imgs.to(DEVICE)
            embeddings = model(batch_imgs)
            embeddings = F.normalize(embeddings, dim=1).cpu().numpy()
            similarities = np.dot(embeddings, ref_embeddings.T)

            for i in range(len(batch_imgs)):
                query_class = batch_class_ids[i]
                query_name = product_map.get(query_class, query_class)
                sample_idx = batch_indices[i].item()

                # --- KNN retrieval (visual) ---
                top_indices = np.argsort(similarities[i])[::-1][:top_k]
                top_classes = [ref_class_names[j] for j in top_indices]
                top_scores = [float(similarities[i][j]) for j in top_indices]
                top_names = [product_map.get(c, c) for c in top_classes]

                total += 1
                per_class_total[query_class] += 1

                # --- Baseline metrics ---
                base_pred_id = top_classes[0]
                base_pred_name = top_names[0]

                if base_pred_id == query_class:
                    base_id[1] += 1
                    per_class_base[query_class] += 1
                if query_class in top_classes[:5]: base_id[5] += 1
                if query_class in top_classes[:10]: base_id[10] += 1
                if base_pred_name == query_name: base_name[1] += 1
                if query_name in top_names[:5]: base_name[5] += 1
                if query_name in top_names[:10]: base_name[10] += 1

                # --- Live OCR on raw crop ---
                crop_pil = dataset._get_crop(sample_idx)
                t0 = time.time()
                ocr_fragments = run_ocr(ocr_reader, crop_pil, ocr_conf)
                ocr_time_total += time.time() - t0

                # --- Fuzzy match OCR text against product names ---
                ocr_matches = {}
                if ocr_fragments:
                    ocr_matches = fuzzy_match_product(ocr_fragments, unique_product_names)

                used_ocr = bool(ocr_matches)
                if used_ocr:
                    ocr_used += 1

                # --- Build boosted scores ---
                # combined = visual_score + weight * fuzzy_score (additive bonus)
                # If no OCR match for a candidate, its score stays unchanged
                boosted = []
                for k in range(len(top_classes)):
                    vis = top_scores[k]
                    cname = top_names[k]
                    ocr_bonus = ocr_matches.get(cname, 0.0)
                    combined = vis + ocr_weight * ocr_bonus
                    boosted.append((top_classes[k], cname, vis, ocr_bonus, combined))

                boosted.sort(key=lambda x: x[4], reverse=True)

                # --- OCR-boosted metrics ---
                ocr_pred_id = boosted[0][0]
                ocr_pred_name = boosted[0][1]

                ocr_pred_id_correct = ocr_pred_id == query_class
                if ocr_pred_id_correct:
                    ocr_id[1] += 1
                    per_class_ocr[query_class] += 1
                ocr_ids = [b[0] for b in boosted]
                ocr_nms = [b[1] for b in boosted]
                if query_class in ocr_ids[:5]: ocr_id[5] += 1
                if query_class in ocr_ids[:10]: ocr_id[10] += 1
                if ocr_pred_name == query_name: ocr_name[1] += 1
                if query_name in ocr_nms[:5]: ocr_name[5] += 1
                if query_name in ocr_nms[:10]: ocr_name[10] += 1

                # Track impact
                base_id_correct = (base_pred_id == query_class)
                base_name_correct = (base_pred_name == query_name)
                ocr_name_correct = (ocr_pred_name == query_name)

                if not base_id_correct and ocr_pred_id_correct: ocr_helped_id += 1
                if base_id_correct and not ocr_pred_id_correct: ocr_hurt_id += 1
                if not base_name_correct and ocr_name_correct: ocr_helped_name += 1
                if base_name_correct and not ocr_name_correct: ocr_hurt_name += 1

                ocr_text = " | ".join(f"{t} ({c:.2f})" for t, c in ocr_fragments) if ocr_fragments else ""
                match_text = ", ".join(f"{n}={s:.2f}" for n, s in ocr_matches.items()) if ocr_matches else ""

                all_results.append({
                    "query_class": query_class,
                    "query_name": query_name,
                    "base_pred_id": base_pred_id,
                    "base_pred_name": base_pred_name,
                    "base_id_correct": base_id_correct,
                    "base_name_correct": base_name_correct,
                    "ocr_pred_id": ocr_pred_id,
                    "ocr_pred_name": ocr_pred_name,
                    "ocr_id_correct": ocr_pred_id_correct,
                    "ocr_name_correct": ocr_name_correct,
                    "ocr_text": ocr_text,
                    "ocr_matches": match_text,
                    "ocr_used": used_ocr,
                    "boosted": boosted[:5],
                })

    metrics = {
        "total": total,
        "base_id": {k: v/total for k, v in base_id.items()},
        "base_name": {k: v/total for k, v in base_name.items()},
        "ocr_id": {k: v/total for k, v in ocr_id.items()},
        "ocr_name": {k: v/total for k, v in ocr_name.items()},
        "ocr_used": ocr_used,
        "ocr_helped_id": ocr_helped_id,
        "ocr_hurt_id": ocr_hurt_id,
        "ocr_helped_name": ocr_helped_name,
        "ocr_hurt_name": ocr_hurt_name,
        "ocr_avg_ms": (ocr_time_total / total * 1000) if total else 0,
        "per_class_base": dict(per_class_base),
        "per_class_ocr": dict(per_class_ocr),
        "per_class_total": dict(per_class_total),
    }
    return metrics, all_results


# =============================================================================
# Main
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--ocr_weight", type=float, default=OCR_WEIGHT)
    parser.add_argument("--ocr_conf", type=float, default=OCR_CONF_THRESHOLD)
    args = parser.parse_args()

    print("=" * 70)
    print("ArcFace + Live OCR Reranking - Migros v6")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Top-K: {args.top_k}, OCR weight: {args.ocr_weight}, OCR conf: {args.ocr_conf}")
    print()

    # Validate
    for p, n in [(TESTING_DIR, "Testing"), (ANNOTATIONS_CSV, "Annotations"),
                  (CHECKPOINT_PATH, "Checkpoint"), (REFERENCE_DB_PATH, "Reference DB")]:
        if not p.exists():
            print(f"ERROR: {n} not found: {p}")
            return
        print(f"  [OK] {n}")

    # Load
    print("\n1. Loading data...")
    product_map = load_product_mapping(PRODUCT_MAP_CSV)
    print(f"  Product mapping: {len(product_map)} entries")
    samples = load_annotations(ANNOTATIONS_CSV, TESTING_DIR)

    unique_product_names = sorted(set(product_map.values()))
    print(f"  Unique product names: {len(unique_product_names)}")

    print("\n2. Loading model...")
    model = load_model(CHECKPOINT_PATH)

    print("\n3. Loading reference DB...")
    ref_embeddings, ref_class_names, _ = load_reference_db(REFERENCE_DB_PATH)

    print("\n4. Initializing OCR...")
    ocr_reader = init_ocr()

    # Dataset
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = CropDataset(samples, test_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Evaluate
    print(f"\n5. Running evaluation...")
    t0 = time.time()
    metrics, results = evaluate(
        model, dataloader, dataset, ref_embeddings, ref_class_names,
        samples, product_map, ocr_reader,
        unique_product_names, args.top_k, args.ocr_weight, args.ocr_conf,
    )
    eval_time = time.time() - t0

    # Print results
    m = metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total crops: {m['total']}")
    print(f"  Eval time: {eval_time:.1f}s  (OCR avg: {m['ocr_avg_ms']:.0f}ms/crop)")
    print(f"  OCR used (found match): {m['ocr_used']}/{m['total']} ({m['ocr_used']*100/m['total']:.0f}%)")
    print()

    print("  +-------------------------------+-----------+-----------+")
    print("  | Metric                        |  Baseline | OCR+KNN   |")
    print("  +-------------------------------+-----------+-----------+")
    print(f"  | Top-1 ID Accuracy             | {m['base_id'][1]*100:8.2f}% | {m['ocr_id'][1]*100:8.2f}% |")
    print(f"  | Top-5 ID Accuracy             | {m['base_id'][5]*100:8.2f}% | {m['ocr_id'][5]*100:8.2f}% |")
    print(f"  | Top-10 ID Accuracy            | {m['base_id'][10]*100:8.2f}% | {m['ocr_id'][10]*100:8.2f}% |")
    print("  +-------------------------------+-----------+-----------+")
    print(f"  | Top-1 Name Accuracy           | {m['base_name'][1]*100:8.2f}% | {m['ocr_name'][1]*100:8.2f}% |")
    print(f"  | Top-5 Name Accuracy           | {m['base_name'][5]*100:8.2f}% | {m['ocr_name'][5]*100:8.2f}% |")
    print(f"  | Top-10 Name Accuracy          | {m['base_name'][10]*100:8.2f}% | {m['ocr_name'][10]*100:8.2f}% |")
    print("  +-------------------------------+-----------+-----------+")

    d_id = (m['ocr_id'][1] - m['base_id'][1]) * 100
    d_name = (m['ocr_name'][1] - m['base_name'][1]) * 100
    print(f"\n  Delta Top-1 ID:   {d_id:+.2f}%")
    print(f"  Delta Top-1 Name: {d_name:+.2f}%")
    print(f"\n  ID:   helped={m['ocr_helped_id']}, hurt={m['ocr_hurt_id']}, net={m['ocr_helped_id']-m['ocr_hurt_id']:+d}")
    print(f"  Name: helped={m['ocr_helped_name']}, hurt={m['ocr_hurt_name']}, net={m['ocr_helped_name']-m['ocr_hurt_name']:+d}")

    # Per-class ID changes
    print("\n" + "-" * 70)
    print("PER-CLASS ID CHANGES (showing classes where OCR changed result)")
    print("-" * 70)
    for cls_id in sorted(m["per_class_total"].keys(), key=int):
        tot = m["per_class_total"][cls_id]
        b = m["per_class_base"].get(cls_id, 0)
        o = m["per_class_ocr"].get(cls_id, 0)
        if b != o:
            name = product_map.get(cls_id, cls_id)
            print(f"  ID {cls_id:>3} {safe(name[:40]):<40} base={b}/{tot}  ocr={o}/{tot}  diff={o-b:+d}")

    # Show OCR examples
    print("\n" + "-" * 70)
    print("SAMPLE OCR MATCHES (crops where OCR found a product match)")
    print("-" * 70)
    shown = 0
    for r in results:
        if r["ocr_used"] and shown < 20:
            gt = safe(r["query_name"])
            pred = safe(r["ocr_pred_name"])
            correct_marker = "[OK]" if r["ocr_name_correct"] else "[WRONG]"
            print(f"  GT: {gt[:35]:<35} OCR: {safe(r['ocr_text'][:50]):<50}")
            print(f"    Matches: {safe(r['ocr_matches'][:70])}")
            print(f"    Pred: {pred} {correct_marker}")
            print()
            shown += 1

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Results file
    with open(OUTPUT_DIR / "ocr_rerank_results.txt", "w", encoding="utf-8") as f:
        f.write("ArcFace + Live OCR Reranking - Migros v6\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {m['total']}, Top-K: {args.top_k}\n")
        f.write(f"OCR weight: {args.ocr_weight}, OCR conf: {args.ocr_conf}\n")
        f.write(f"OCR used: {m['ocr_used']}/{m['total']}\n\n")

        f.write("--- ID Accuracy ---\n")
        f.write(f"Baseline Top-1: {m['base_id'][1]*100:.2f}%\n")
        f.write(f"OCR+KNN  Top-1: {m['ocr_id'][1]*100:.2f}%\n")
        f.write(f"Delta: {d_id:+.2f}%\n\n")

        f.write("--- Name Accuracy ---\n")
        f.write(f"Baseline Top-1: {m['base_name'][1]*100:.2f}%\n")
        f.write(f"OCR+KNN  Top-1: {m['ocr_name'][1]*100:.2f}%\n")
        f.write(f"Delta: {d_name:+.2f}%\n\n")

        f.write(f"ID:   helped={m['ocr_helped_id']}, hurt={m['ocr_hurt_id']}\n")
        f.write(f"Name: helped={m['ocr_helped_name']}, hurt={m['ocr_hurt_name']}\n")

    # Per-crop CSV
    import csv
    with open(OUTPUT_DIR / "per_crop_details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_class", "query_name", "base_pred_id", "base_pred_name",
                         "base_id_ok", "base_name_ok", "ocr_pred_id", "ocr_pred_name",
                         "ocr_id_ok", "ocr_name_ok", "ocr_text", "ocr_matches", "ocr_used"])
        for r in results:
            writer.writerow([
                r["query_class"], r["query_name"],
                r["base_pred_id"], r["base_pred_name"],
                r["base_id_correct"], r["base_name_correct"],
                r["ocr_pred_id"], r["ocr_pred_name"],
                r["ocr_id_correct"], r["ocr_name_correct"],
                r["ocr_text"], r["ocr_matches"], r["ocr_used"],
            ])

    print(f"\n  Results: {OUTPUT_DIR / 'ocr_rerank_results.txt'}")
    print(f"  Details: {OUTPUT_DIR / 'per_crop_details.csv'}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
